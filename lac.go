package lac

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/go-aie/paddle"
	aietokenizer "github.com/go-aie/tokenizer"
	"github.com/sugarme/tokenizer"
)

type Config struct {
	ModelPath, ParamsPath string
	WordVocabFile         string
	TagVocabFile          string
	Q2bVocabFile          string
	UserDict              string
	DoLowerCase           bool
	ForCN                 bool
	// The maximum number of predictors for concurrent inferences.
	// Defaults to the value of runtime.NumCPU.
	MaxConcurrency int
}

type LAC struct {
	engine *paddle.Engine

	tk       *aietokenizer.Tokenizer
	tagVocab *aietokenizer.Vocab[int64]
	custom   *Customization
}

func NewLAC(cfg *Config) *LAC {
	tk, err := NewTokenizer(cfg.WordVocabFile, cfg.Q2bVocabFile)
	if err != nil {
		panic(err)
	}

	tagVocab, err := aietokenizer.NewVocabFromFile[int64](cfg.TagVocabFile, "\t", "O")
	if err != nil {
		panic(err)
	}

	custom := NewCustomization()
	if cfg.UserDict != "" {
		if err := custom.LoadFromFile(cfg.UserDict); err != nil {
			panic(err)
		}
	}

	return &LAC{
		engine:   paddle.NewEngine(cfg.ModelPath, cfg.ParamsPath, cfg.MaxConcurrency),
		tk:       tk,
		tagVocab: tagVocab,
		custom:   custom,
	}
}

func (l *LAC) LAC(texts []string) ([]Segments, error) {
	encodings, err := l.tk.EncodeBatchTexts(texts, false)
	if err != nil {
		return nil, err
	}

	inputs, lens := l.getInputs(encodings)
	outputs := l.engine.Infer(inputs)
	result := outputs[0]

	rows := paddle.NewMatrix[int64](result).Rows()

	var dataSet []Data
	for i, row := range rows {
		dataSet = append(dataSet, Data{
			Text:   texts[i],
			TagIDs: row[:lens[i]], // Remove the padding ids.
		})
	}
	return l.postProcess(dataSet), nil
}

func (l *LAC) Seg(texts []string) ([][]string, error) {
	result, err := l.LAC(texts)
	if err != nil {
		return nil, err
	}

	var words [][]string
	for _, s := range result {
		words = append(words, s.Words())
	}
	return words, nil
}

func (l *LAC) getInputs(encodings []tokenizer.Encoding) ([]paddle.Tensor, []int64) {
	var ids [][]int64
	var lens []int64

	for _, e := range encodings {
		ids = append(ids, paddle.NumberToInt64(e.Ids))

		// Calculate the length of the encoded tokens, not counting the padding ones.
		length := 0
		for ; length < len(e.AttentionMask) && e.AttentionMask[length] == 1; length++ {
		}
		lens = append(lens, int64(length))
	}

	return []paddle.Tensor{
		paddle.NewInputTensorFromTwoDimSlice(ids),
		paddle.NewInputTensorFromOneDimSlice(lens),
	}, lens
}

func (l *LAC) postProcess(dataSet []Data) []Segments {
	var result []Segments
	for _, d := range dataSet {
		var tags []*Tag
		tokens, err := l.tagVocab.IDsToTokens(d.TagIDs)
		if err != nil {
			panic(err)
		}
		for _, s := range tokens {
			tags = append(tags, NewTag(s))
		}
		fmt.Printf("tags: %v\n", tags)

		runes := []rune(d.Text)
		l.custom.Parse(d.Text, tags)

		var word string
		var end int
		var segments Segments
		var prevTag *Tag

		for i, tag := range tags {
			if tag.BIO == "B" || (tag.BIO == "O" && prevTag.BIO != "O") {
				if word != "" {
					segments = append(segments, Segment{
						Word: word,
						POS:  prevTag.POS,
						Offset: Offset{
							Start: end - utf8.RuneCountInString(word),
							End:   end,
						},
					})
					word = ""
				}
			}

			word += string(runes[i])
			end++
			prevTag = tag
		}

		if word != "" {
			segments = append(segments, Segment{
				Word: word,
				POS:  prevTag.POS,
				Offset: Offset{
					Start: end - utf8.RuneCountInString(word),
					End:   end,
				},
			})
		}

		result = append(result, segments)
	}
	return result
}

type Offset struct {
	Start int // The start offset.
	End   int // The end offset.
}

type Segment struct {
	Word string
	POS  string
	Offset
}

type Segments []Segment

func (s Segments) Words() []string {
	var result []string
	for _, seg := range s {
		result = append(result, seg.Word)
	}
	return result
}

func (s Segments) POSs() []string {
	var result []string
	for _, seg := range s {
		result = append(result, seg.POS)
	}
	return result
}

type Data struct {
	Text   string
	TagIDs []int64
}

type Tag struct {
	// Part-of-speech tag.
	// See https://github.com/baidu/lac#%E8%AF%8D%E6%80%A7%E6%A0%87%E6%B3%A8%E4%B8%8E%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB.
	POS string
	// Inside–outside–beginning tag.
	// See https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging).
	BIO string
}

func NewTag(s string) *Tag {
	sub := strings.SplitN(s, "-", 2)
	pos, bio := sub[0], ""
	if len(sub) == 2 {
		bio = sub[1]
	}
	return &Tag{
		POS: pos,
		BIO: bio,
	}
}

func (t *Tag) String() string {
	if t.BIO == "" {
		return t.POS
	}
	return t.POS + "-" + t.BIO
}

func (t *Tag) SetPOSIfNonempty(pos string) {
	if pos != "" {
		t.POS = pos
	}
}
