package lac

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	aietokenizer "github.com/go-aie/tokenizer"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/model/wordlevel"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
)

func NewTokenizer(wordVocabFile, q2bVocabFile string) (*aietokenizer.Tokenizer, error) {
	m, err := NewWordLevel(wordVocabFile, q2bVocabFile)
	if err != nil {
		return nil, err
	}

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

type WordLevel struct {
	*wordlevel.WordLevel
	q2bVocab map[string]string
}

func NewWordLevel(wordVocabFile, q2bVocabFile string) (*WordLevel, error) {
	wordVocab, err := aietokenizer.NewVocabFromFile[int](wordVocabFile, "\t")
	if err != nil {
		return nil, err
	}

	q2bVocab, err := LoadQ2bVocab(q2bVocabFile)
	if err != nil {
		return nil, err
	}

	return &WordLevel{
		WordLevel: aietokenizer.NewWordLevel(wordVocab.Vocab, "OOV"),
		q2bVocab:  q2bVocab,
	}, nil
}

// Tokenize transforms given input into token.
func (w *WordLevel) Tokenize(token string) ([]tokenizer.Token, error) {
	// Convert traditional tokens to simplified tokens.
	if t, ok := w.q2bVocab[token]; ok {
		token = t
	}
	return w.WordLevel.Tokenize(token)
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
