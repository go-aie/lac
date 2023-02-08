package lac

import (
	"bufio"
	"os"
	"strings"
	"unicode/utf8"
)

type Customization struct {
	items map[string][]Segment
	tree  Trie
}

func NewCustomization() *Customization {
	return &Customization{
		items: make(map[string][]Segment),
		tree:  NewTrie(),
	}
}

func (c *Customization) LoadFromFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	var lines []string

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	c.LoadFromSlice(lines)
	return nil
}

func (c *Customization) LoadFromSlice(lines []string) {
	for _, line := range lines {
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}

		var word string
		var segments []Segment

		for _, field := range fields {
			w, pos := field, ""
			if sub := strings.SplitN(field, "/", 2); len(sub) == 2 {
				w, pos = sub[0], sub[1]
			}

			start := utf8.RuneCountInString(word)
			segments = append(segments, Segment{
				Word: w,
				POS:  pos,
				Offset: Offset{
					Start: start,
					End:   start + utf8.RuneCountInString(w),
				},
			})

			word += w
		}

		c.items[word] = segments
		c.tree.Add(word)
	}
}

func (c *Customization) Parse(text string, tags []*Tag) {
	for _, offset := range c.tree.Search(text) {
		start, end := offset.Start, offset.End
		word := string([]rune(text)[start:end])

		for _, segment := range c.items[word] {
			// For the first character, change its tag name and mark it as "B".
			i := start + segment.Start
			tags[i].SetPOSIfNonempty(segment.POS)
			tags[i].BIO = "B"

			// For the intermediate characters, change their tag names and mark them as "I".
			for i = i + 1; i < start+segment.End; i++ {
				tags[i].SetPOSIfNonempty(segment.POS)
				tags[i].BIO = "I"
			}
		}

		// Mark the character immediately next to this word as "B".
		if end < len(tags) {
			tags[end].BIO = "B"
		}
	}
}

type Trie struct {
	tree map[string]int
}

func NewTrie() Trie {
	return Trie{tree: make(map[string]int)}
}

func (t Trie) Add(word string) {
	if word == "" {
		return
	}

	runes := []rune(word)
	t.tree[word] = len(runes)

	for i := 1; i < len(runes); i++ {
		frag := string(runes[:i])
		if _, ok := t.tree[frag]; !ok {
			t.tree[frag] = 0
		}
	}
}

// Search implements FMM (Forward maximum matching).
func (t Trie) Search(text string) []Offset {
	var offsets []Offset

	runes := []rune(text)
	for start := 0; start < len(runes); start++ {
		for end := start + 1; end < len(runes)+1; end++ {
			v, ok := t.tree[string(runes[start:end])]
			if !ok {
				break
			}
			if v > 0 && (len(offsets) == 0 || end > offsets[len(offsets)-1].End) {
				offsets = append(offsets, Offset{Start: start, End: end})
			}
		}
	}

	return offsets
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
