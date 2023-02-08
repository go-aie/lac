package lac_test

import (
	"testing"

	"github.com/go-aie/lac"
	"github.com/google/go-cmp/cmp"
)

func TestLAC_Seg(t *testing.T) {
	l := newLAC()

	tests := []struct {
		inText     []string
		inUserDict string
		wantWords  [][]string
	}{
		{
			inText: []string{
				"近日国家卫健委发布第九版新型冠状病毒肺炎诊疗方案",
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
