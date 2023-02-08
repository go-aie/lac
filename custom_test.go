package lac_test

import (
	"testing"

	"github.com/go-aie/lac"
	"github.com/google/go-cmp/cmp"
)

func TestCustomization_Parse(t *testing.T) {
	tests := []struct {
		inText         string
		inTagStrings   []string
		inUserDict     []string
		wantTagStrings []string
	}{
		{
			inText:       "春天的花开秋天的风以及冬天的落阳",
			inTagStrings: []string{"TIME-B", "TIME-I", "u-B", "n-B", "v-I", "TIME-B", "TIME-I", "u-B", "n-B", "c-B", "c-I", "TIME-B", "TIME-I", "u-B", "LOC-B", "PER-I"},
			inUserDict: []string{
				"春天/SEASON",
				"花/n 开/v",
				"秋天的风",
				"落 阳",
			},
			wantTagStrings: []string{"SEASON-B", "SEASON-I", "u-B", "n-B", "v-B", "TIME-B", "TIME-I", "u-I", "n-I", "c-B", "c-I", "TIME-B", "TIME-I", "u-B", "LOC-B", "PER-B"},
		},
	}
	for _, tt := range tests {
		custom := lac.NewCustomization()
		custom.LoadFromSlice(tt.inUserDict)

		var inTags []*lac.Tag
		for _, s := range tt.inTagStrings {
			inTags = append(inTags, lac.NewTag(s))
		}

		custom.Parse(tt.inText, inTags)

		var gotTagStrings []string
		for _, t := range inTags {
			gotTagStrings = append(gotTagStrings, t.String())
		}

		if !cmp.Equal(gotTagStrings, tt.wantTagStrings) {
			diff := cmp.Diff(gotTagStrings, tt.wantTagStrings)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}

func TestTriedTree_Search(t *testing.T) {
	tests := []struct {
		inWords     []string
		inText      string
		wantOffsets []lac.Offset
	}{
		{
			inWords: []string{"春天", "花开", "秋天的风", "落阳"},
			inText:  "春天的花开秋天的风以及冬天的落阳",
			wantOffsets: []lac.Offset{
				{Start: 0, End: 2},   // 春天
				{Start: 3, End: 5},   // 花开
				{Start: 5, End: 9},   // 秋天的风
				{Start: 14, End: 16}, // 洛阳
			},
		},
	}
	for _, tt := range tests {
		tree := lac.NewTrie()
		for _, word := range tt.inWords {
			tree.Add(word)
		}
		gotResult := tree.Search(tt.inText)
		if !cmp.Equal(gotResult, tt.wantOffsets) {
			diff := cmp.Diff(gotResult, tt.wantOffsets)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}
