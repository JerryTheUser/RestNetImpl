# RestNetImpl
- 利用PyTorch手刻 ResNet18/ResNet50 模型
- 引用 torchvishion 裡面預先訓練好的模型來比較
- 外加一個自製 data loader,在每次需要資料的時候載入資料
- RestNet family:
  - ![image](https://github.com/JerryTheUser/RestNetImpl/blob/main/img/arch.png)

## Hyper-parameters
- Batch Size : 4
- Epoch : 10(Res18) / 3(Res50)
- OPtimizer : SGD
- Learning Rate : 0.001
- Loss Function : CrossEntropy

## Results
![image](https://github.com/JerryTheUser/RestNetImpl/blob/main/img/res18.png)
