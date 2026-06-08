代码文件包括： 
  （1）utils：数据预处理
  （2）trainer：训练，checkpoint用于暂停训练
  （3）models：模型，attention_branch是transformer模型，cnn_branch是cnn模型，cnn_branch是多模态融合模型
  （4）config：配置文件
  （5）带main，plot和save开头的文件为接口文件和画图文件：main_build_dataset为数据预处理接口，main_test和main_train分别为测试集和训练集，其他为画图文件

  checkpoint保存了训练的迭代变化曲线和模型训练过程中的最佳参数组合
  compare_plots包含了对比试验和消融实验的图片
  logs是训练日志，保存了每轮训练的模型评价指标
