package lac_test

import (
	"testing"

	"github.com/go-aie/lac"
	"github.com/google/go-cmp/cmp"
)

func TestLAC_LAC(t *testing.T) {
	l := newLAC()

	tests := []struct {
		inText       []string
		wantSegments []lac.Segments
	}{
		{
			inText: []string{
				"近日国家卫健委发布第九版新型冠状病毒肺炎诊疗方案",
				"LAC是个优秀的分词工具",
				"我的想法",
			},
			wantSegments: []lac.Segments{
				{
					{Word: "近日", POS: "TIME", Offset: lac.Offset{End: 2}},
					{Word: "国家卫健委", POS: "ORG", Offset: lac.Offset{Start: 2, End: 7}},
					{Word: "发布", POS: "v", Offset: lac.Offset{Start: 7, End: 9}},
					{Word: "第九版", POS: "m", Offset: lac.Offset{Start: 9, End: 12}},
					{Word: "新型", POS: "a", Offset: lac.Offset{Start: 12, End: 14}},
					{Word: "冠状病毒肺炎", POS: "nz", Offset: lac.Offset{Start: 14, End: 20}},
					{Word: "诊疗", POS: "vn", Offset: lac.Offset{Start: 20, End: 22}},
					{Word: "方案", POS: "n", Offset: lac.Offset{Start: 22, End: 24}},
				},
				{
					{Word: "LAC", POS: "nz", Offset: lac.Offset{End: 3}},
					{Word: "是", POS: "v", Offset: lac.Offset{Start: 3, End: 4}},
					{Word: "个", POS: "q", Offset: lac.Offset{Start: 4, End: 5}},
					{Word: "优秀", POS: "a", Offset: lac.Offset{Start: 5, End: 7}},
					{Word: "的", POS: "u", Offset: lac.Offset{Start: 7, End: 8}},
					{Word: "分词", POS: "n", Offset: lac.Offset{Start: 8, End: 10}},
					{Word: "工具", POS: "n", Offset: lac.Offset{Start: 10, End: 12}},
				},
				{
					{Word: "我", POS: "r", Offset: lac.Offset{End: 1}},
					{Word: "的", POS: "u", Offset: lac.Offset{Start: 1, End: 2}},
					{Word: "想法", POS: "n", Offset: lac.Offset{Start: 2, End: 4}},
				},
			},
		},
	}
	for _, tt := range tests {
		gotSegments, err := l.LAC(tt.inText)
		if err != nil {
			t.Errorf("err: %v\n", err)
		}
		if !cmp.Equal(gotSegments, tt.wantSegments) {
			diff := cmp.Diff(gotSegments, tt.wantSegments)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}

func TestLAC_Seg(t *testing.T) {
	l := newLAC()

	tests := []struct {
		inText    []string
		wantWords [][]string
	}{
		{
			inText: []string{
				"近日国家卫健委发布第九版新型冠状病毒肺炎诊疗方案",
				"LAC是个优秀的分词工具",
				"我的想法",
			},
			wantWords: [][]string{
				{
					"近日",
					"国家卫健委",
					"发布",
					"第九版",
					"新型",
					"冠状病毒肺炎",
					"诊疗",
					"方案",
				},
				{
					"LAC",
					"是",
					"个",
					"优秀",
					"的",
					"分词",
					"工具",
				},
				{
					"我",
					"的",
					"想法",
				},
			},
		},
	}
	for _, tt := range tests {
		gotWords, err := l.Seg(tt.inText)
		if err != nil {
			t.Errorf("err: %v\n", err)
		}
		if !cmp.Equal(gotWords, tt.wantWords) {
			diff := cmp.Diff(gotWords, tt.wantWords)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}

func newLAC() *lac.LAC {
	return lac.NewLAC(&lac.Config{
		ModelPath:     "./lac/static/inference.pdmodel",
		ParamsPath:    "./lac/static/inference.pdiparams",
		WordVocabFile: "./lac/word.dic",
		TagVocabFile:  "./lac/tag.dic",
		Q2bVocabFile:  "./lac/q2b.dic",
	})
}
