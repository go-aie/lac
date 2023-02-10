package lac

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	aietokenizer "github.com/go-aie/tokenizer"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
)

func NewTokenizer(wordVocabFile, q2bVocabFile string) (*aietokenizer.Tokenizer, error) {
	vocab, err := NewVocab(wordVocabFile, q2bVocabFile)
	if err != nil {
		return nil, err
	}
	m := aietokenizer.NewRuneLevel(vocab)

	paddingStrategy := tokenizer.NewPaddingStrategy()
	paddingParams := tokenizer.PaddingParams{
		Strategy:  *paddingStrategy,
		Direction: tokenizer.Right, // padding right
	}
	tk := tokenizer.NewTokenizer(m)
	tk.WithPadding(&paddingParams)
	tk.WithNormalizer(normalizer.NewBertNormalizer(false, true, true, false)) // Handle Chinese chars
	tk.WithPreTokenizer(pretokenizer.NewBertPreTokenizer())

	return &aietokenizer.Tokenizer{Tokenizer: tk}, nil
}

type Vocab struct {
	wordVocab *aietokenizer.Vocab[int]
	q2bVocab  map[string]string
}

func NewVocab(wordVocabFile, q2bVocabFile string) (*Vocab, error) {
	wordVocab, err := aietokenizer.NewVocabFromFile[int](wordVocabFile, "\t", "OOV")
	if err != nil {
		return nil, err
	}

	q2bVocab, err := LoadQ2bVocab(q2bVocabFile)
	if err != nil {
		return nil, err
	}

	return &Vocab{
		wordVocab: wordVocab,
		q2bVocab:  q2bVocab,
	}, nil
}

func (v *Vocab) Vocab() map[string]int { return v.wordVocab.Vocab() }
func (v *Vocab) UnkToken() string      { return v.wordVocab.UnkToken() }

func (v *Vocab) TokenToID(token string) (int, error) {
	// Convert traditional tokens to simplified tokens.
	if t, ok := v.q2bVocab[token]; ok {
		token = t
	}
	return v.wordVocab.TokenToID(token)
}

func LoadQ2bVocab(filename string) (map[string]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	vocab := make(map[string]string)

	scanner := bufio.NewScanner(file)
	for i := 0; scanner.Scan(); i++ {
		line := scanner.Text()
		sub := strings.Split(line, "\t")
		if len(sub) != 2 {
			return nil, fmt.Errorf("invalid content: %q at line %d", line, i+1)
		}

		vocab[sub[0]] = sub[1]
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return vocab, nil
}
