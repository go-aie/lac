package lac

import (
	"fmt"
	"strings"

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

	tagVocab, err := aietokenizer.NewVocabFromFile[int64](cfg.TagVocabFile, "\t")
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

func (l *LAC) Seg(texts []string) ([][]string, error) {
	encodings, err := l.tk.EncodeBatchTexts(texts, false)
	if err != nil {
		return nil, err
	}

	inputs, lens := l.getInputs(encodings)
	outputs := l.engine.Infer(inputs)
	result := outputs[0]

	m := paddle.NewMatrix(paddle.NewTypedTensor[int64](result))
	rows := paddle.Rows[int64](m)

	var dataSet []Data
	for i, row := range rows {
		dataSet = append(dataSet, Data{
			Text:   texts[i],
			TagIDs: row[:lens[i]], // Remove the padding ids.
		})
	}

	return l.postProcess(dataSet), nil
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

func (l *LAC) postProcess(dataSet []Data) [][]string {
	var result [][]string
	for _, d := range dataSet {
		var tags []*Tag
		for _, s := range l.tagVocab.IDsToTokens(d.TagIDs) {
			tags = append(tags, NewTag(s))
		}
		fmt.Printf("tags: %v\n", tags)

		runes := []rune(d.Text)
		l.custom.Parse(d.Text, tags)

		var words []string
		word := ""
		var prevTag *Tag
		for i, tag := range tags {
			if tag.BIO == "B" || (tag.BIO == "O" && prevTag.BIO != "O") {
				if word != "" {
					words = append(words, word)
					word = ""
				}
			}

			word += string(runes[i])
			prevTag = tag
		}

		if word != "" {
			words = append(words, word)
		}

		result = append(result, words)
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
