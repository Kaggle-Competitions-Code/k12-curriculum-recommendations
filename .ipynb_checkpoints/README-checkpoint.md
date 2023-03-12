# k12-curriculum-recommendations
[Learning Equality - Curriculum Recommendations](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/)

# Stage 1 - Retrieval
- CV思路: 
    - source category作为所有fold训练集的一部分, 不参与验证集的本地验证. stratifiedgroupkfold非source category的topic, group=topic, target=language.
    - topic, content不直接split成train valid数据集, 而是通过split correlations进行train valid set的划分, 训练集/测试集会有重复的content, topic tree, 预期会造成一定的数据泄露, 但实测CV/LB的涨势一致符合预期
- 预训练模型: sentence-transformers/all-MiniLM-L6-v2
    - 参数量小, 相对来讲比较容易做实验
    - 依据讨论区分享, 更换大模型预期会带来较大的提升, 但由于时间原因, 没有尝试过使用更大模型
- 训练思路: SimCSE对比学习
    - groupby language, then shuffle 使每个batch里尽量只有一种语言能提升不到1的召回
    - 使用hard negative能提升模型召回
    - 使用旧模型召回topN sample负样本boost旧模型能提升模型召回
    - 仅使用非source的数据boost旧模型2轮能提升top50/100的召回, 但从第三轮起没有提升
- 模型融合: 
    - 投票 + topic&content pair平均cosine距离从小到大排序

## 提分点
- 负样本
- non-source topic很关键, 因为其distribution与LB测试集一致
- 多轮boost

## 反思
- 拖到比赛结束前20天才真正开始打比赛, 导致一阶段很匆促, 二阶段也很被动
- 没有讨论协调一致一二阶段的CV, 导致二阶段验证会出现一些数据泄露的问题
- 一阶段看到讨论区大佬使用小模型的表现好以致于太执着于小模型, 只想着增加maxlen, batchsize以增加模型召回, 没有尝试通过减小maxlen, batchsize以训练更大的模型提升召回表现. 实测maxlen128, batchsize192以后再增加其长度提升很微妙
- 一开始cv没有做好, 导致模型融合以后没有验证集来衡量融合模型的表现, 需要重做cv. 
- 可以考虑将更多的non source topic分入训练集, 来给模型更多高质量的样本
- 代码水平有限, 没想到怎样通过魔改simcse增加正负样本比