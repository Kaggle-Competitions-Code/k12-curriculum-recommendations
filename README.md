# k12-curriculum-recommendations
[Learning Equality - Curriculum Recommendations](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/)

训练推理代码，全部在`src`中，其他中途实验参考 `git log`

## Stage2
采用二分类模型（softmax+CEloss），样本不均衡问题采用利用阈值分割。
* 不选择sigmoid+MSEloss的原因——MSEloss相比于CEloss是非凸的，具体实验效果未尝试
* 对应样本不均衡没加入Focal Loss的原因——资源不足，只能采用MP训练，当时对于Loss在MP的切分不确定，没能修改（已复盘学习实验）
* tokenizer采用pair的形式来保证text1和text2长度一致，并复用一阶段的prompt
* 因推荐中全链路一致的情况下，召回order是个不错的特征，将特征文本化加入prompt，有提升，但很小
* 继续预训练未带来提高
* 更换更大的模型明显带来更大的提高，时间有限，尝试的不多
* CV采用和LB完全一致的策略分割


## 复盘
* CV方案和一阶段不一致，导致存在泄露，应该果断切换一阶段数据消融实验
* 在有限时间内，对于不同策略及双阶段融合时间把握还可以，在最后完成了想做的
* 平时的相关实验可以多做一些，时间有限的情况下，可以在不实验的情况下，判断出方案的优先级（比较重要）
* 在比赛后的这段时间，一直在更深入的学习和实验分布式相关的内容，保证可以在有限的资源数值上，计算出自己能选取模型大小的上限


## 最后
相比于匆匆忙忙的上个otto比赛，已经在有限时间内，完成的比较好了，感谢队友