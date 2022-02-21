# Deepfakes classification using U-Net backbone

## Model Architecture

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1         [-1, 64, 320, 320]           1,792
       BatchNorm2d-2         [-1, 64, 320, 320]             128
              ReLU-3         [-1, 64, 320, 320]               0
            Conv2d-4         [-1, 64, 320, 320]          36,928
       BatchNorm2d-5         [-1, 64, 320, 320]             128
              ReLU-6         [-1, 64, 320, 320]               0
        DoubleConv-7         [-1, 64, 320, 320]               0
         MaxPool2d-8         [-1, 64, 160, 160]               0
            Conv2d-9        [-1, 128, 160, 160]          73,856
      BatchNorm2d-10        [-1, 128, 160, 160]             256
             ReLU-11        [-1, 128, 160, 160]               0
           Conv2d-12        [-1, 128, 160, 160]         147,584
      BatchNorm2d-13        [-1, 128, 160, 160]             256
             ReLU-14        [-1, 128, 160, 160]               0
       DoubleConv-15        [-1, 128, 160, 160]               0
             Down-16        [-1, 128, 160, 160]               0
        MaxPool2d-17          [-1, 128, 80, 80]               0
           Conv2d-18          [-1, 256, 80, 80]         295,168
      BatchNorm2d-19          [-1, 256, 80, 80]             512
             ReLU-20          [-1, 256, 80, 80]               0
           Conv2d-21          [-1, 256, 80, 80]         590,080
      BatchNorm2d-22          [-1, 256, 80, 80]             512
             ReLU-23          [-1, 256, 80, 80]               0
       DoubleConv-24          [-1, 256, 80, 80]               0
             Down-25          [-1, 256, 80, 80]               0
        MaxPool2d-26          [-1, 256, 40, 40]               0
           Conv2d-27          [-1, 512, 40, 40]       1,180,160
      BatchNorm2d-28          [-1, 512, 40, 40]           1,024
             ReLU-29          [-1, 512, 40, 40]               0
           Conv2d-30          [-1, 512, 40, 40]       2,359,808
      BatchNorm2d-31          [-1, 512, 40, 40]           1,024
             ReLU-32          [-1, 512, 40, 40]               0
       DoubleConv-33          [-1, 512, 40, 40]               0
             Down-34          [-1, 512, 40, 40]               0
        MaxPool2d-35          [-1, 512, 20, 20]               0
           Conv2d-36         [-1, 1024, 20, 20]       4,719,616
      BatchNorm2d-37         [-1, 1024, 20, 20]           2,048
             ReLU-38         [-1, 1024, 20, 20]               0
           Conv2d-39         [-1, 1024, 20, 20]       9,438,208
      BatchNorm2d-40         [-1, 1024, 20, 20]           2,048
             ReLU-41         [-1, 1024, 20, 20]               0
       DoubleConv-42         [-1, 1024, 20, 20]               0
             Down-43         [-1, 1024, 20, 20]               0
        MaxPool2d-44         [-1, 1024, 10, 10]               0
           Conv2d-45         [-1, 2048, 10, 10]      18,876,416
      BatchNorm2d-46         [-1, 2048, 10, 10]           4,096
             ReLU-47         [-1, 2048, 10, 10]               0
           Conv2d-48         [-1, 2048, 10, 10]      37,750,784
      BatchNorm2d-49         [-1, 2048, 10, 10]           4,096
             ReLU-50         [-1, 2048, 10, 10]               0
       DoubleConv-51         [-1, 2048, 10, 10]               0
             Down-52         [-1, 2048, 10, 10]               0
           Linear-53                 [-1, 1024]       2,098,176
           Linear-54                  [-1, 512]         524,800
           Linear-55                    [-1, 1]             513
  ConvTranspose2d-56          [-1, 512, 40, 40]       2,097,664
           Conv2d-57          [-1, 512, 40, 40]       4,719,104
      BatchNorm2d-58          [-1, 512, 40, 40]           1,024
             ReLU-59          [-1, 512, 40, 40]               0
           Conv2d-60          [-1, 512, 40, 40]       2,359,808
      BatchNorm2d-61          [-1, 512, 40, 40]           1,024
             ReLU-62          [-1, 512, 40, 40]               0
       DoubleConv-63          [-1, 512, 40, 40]               0
               Up-64          [-1, 512, 40, 40]               0
  ConvTranspose2d-65          [-1, 256, 80, 80]         524,544
           Conv2d-66          [-1, 256, 80, 80]       1,179,904
      BatchNorm2d-67          [-1, 256, 80, 80]             512
             ReLU-68          [-1, 256, 80, 80]               0
           Conv2d-69          [-1, 256, 80, 80]         590,080
      BatchNorm2d-70          [-1, 256, 80, 80]             512
             ReLU-71          [-1, 256, 80, 80]               0
       DoubleConv-72          [-1, 256, 80, 80]               0
               Up-73          [-1, 256, 80, 80]               0
  ConvTranspose2d-74        [-1, 128, 160, 160]         131,200
           Conv2d-75        [-1, 128, 160, 160]         295,040
      BatchNorm2d-76        [-1, 128, 160, 160]             256
             ReLU-77        [-1, 128, 160, 160]               0
           Conv2d-78        [-1, 128, 160, 160]         147,584
      BatchNorm2d-79        [-1, 128, 160, 160]             256
             ReLU-80        [-1, 128, 160, 160]               0
       DoubleConv-81        [-1, 128, 160, 160]               0
               Up-82        [-1, 128, 160, 160]               0
  ConvTranspose2d-83         [-1, 64, 320, 320]          32,832
           Conv2d-84         [-1, 64, 320, 320]          73,792
      BatchNorm2d-85         [-1, 64, 320, 320]             128
             ReLU-86         [-1, 64, 320, 320]               0
           Conv2d-87         [-1, 64, 320, 320]          36,928
      BatchNorm2d-88         [-1, 64, 320, 320]             128
             ReLU-89         [-1, 64, 320, 320]               0
       DoubleConv-90         [-1, 64, 320, 320]               0
               Up-91         [-1, 64, 320, 320]               0
           Conv2d-92          [-1, 2, 320, 320]             130
          OutConv-93          [-1, 2, 320, 320]               0

Total params: 90,302,467
Trainable params: 90,302,467
Non-trainable params: 0

Input size (MB): 1.17
Forward/backward pass size (MB): 1608.61
Params size (MB): 344.48
Estimated Total Size (MB): 1954.25

- To get predictions on a frame, put weights from [drive link](https://drive.google.com/drive/folders/1ST3ygCUMGP4tyQz8T7NezUgKFhXt5CRk?usp=sharing) in checkpoints folder and run `fyp_predictions.ipynb`

- To get accuracy of the model, put weights from [drive link](https://drive.google.com/drive/folders/1ST3ygCUMGP4tyQz8T7NezUgKFhXt5CRk?usp=sharing) in checkpoints folder and run `fyp_results.py`
