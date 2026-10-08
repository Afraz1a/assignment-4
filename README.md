# Fine-tuning ResNet-50 and U-Net on Small Image Datasets

**ANN & Deep Learning - Assignment 02**
**Name:** afrazia | **Roll No.:** bscs23029

In this assignment I fine-tuned two pretrained models on very small datasets:

| Part | Dataset | Model | Task |
|---|---|---|---|
| A | Oxford Flowers-102 (5 classes x 40 images = 200) | ResNet-50 | Classification |
| B | Oxford-IIIT Pet (150 image + mask pairs) | U-Net (ResNet-50 encoder) | Segmentation |

## Final Results

| ResNet-50 | Result | U-Net | Result |
|---|---:|---|---:|
| Training accuracy | 1.0000 | Training loss | 0.1145 |
| Validation accuracy | 0.9667 | Validation loss | 0.3272 |
| Test accuracy | 1.0000 | Test Dice | 0.9303 |
| Test F1 (macro) | 1.0000 | Test IoU | 0.8752 |

## How I Did Each Task

### Part A - ResNet-50 (Classification)
1. **Task 1:** Counted the images in each class folder with `os.listdir`, printed the totals (5 classes, 200 images) and drew a bar chart.
2. **Task 2:** Opened the first image of every class and showed it with its class name and size.
3. **Task 3:** Made `train_tf` (resize, random flip, rotation, colour jitter, ToTensor, Normalize) and `eval_tf` (resize, ToTensor, Normalize, no augmentation).
4. **Task 4:** Split the data 70/15/15 using `train_test_split` with `stratify`, made the four Subsets and the DataLoaders (batch size 16).
5. **Task 5:** Replaced the last layer with `nn.Linear(2048, 5)`.
6. **Task 6:** Wrote `run_epoch_cls` (forward pass, loss, backward and optimizer step when training, count loss and correct predictions) and used `CrossEntropyLoss`.
7. **Task 7 (Phase 1):** Froze the whole backbone, trained only `fc` with AdamW (lr 1e-3). Trainable parameters: 10,245.
8. **Task 8 (Phase 2):** Unfroze `layer4` and trained again with a smaller lr (1e-4). Trainable parameters: 14,974,981.
9. **Task 9:** Predicted on train, validation and test sets and calculated accuracy and macro precision, recall and F1.
10. **Task 10:** Made the confusion matrix, saved it as a picture and found the class with the lowest accuracy.
11. **Task 11:** Looked for wrong test predictions. There were none, so nothing was shown.

### Part B - U-Net (Segmentation)
12. **Task 12:** Matched every image with its mask by file name (150 valid pairs) and showed 4 samples.
13. **Task 13:** Wrote `PetSegDataset`. Image and mask are resized together, and flipped and rotated with the same random values. The mask uses NEAREST resizing so it stays 0/1.
14. **Task 14:** Split the pairs 70/15/15 (105/22/23) and made the DataLoaders (batch size 8).
15. **Task 15:** Wrote `dice_loss`, `seg_loss` (BCE + Dice), `dice_score` and `iou_score`.
16. **Task 16:** Wrote `run_epoch_seg` that returns the average loss, Dice and IoU.
17. **Task 17 (Phase 1):** Froze the encoder and trained the decoder and head (lr 1e-3). Trainable parameters: 9,013,073.
18. **Task 18 (Phase 2):** Unfroze `layer3` and `layer4` and used two learning rates: 1e-4 for the encoder and 3e-4 for the rest.
19. **Task 19:** Wrote `evaluate_seg` to get loss, Dice, IoU and pixel accuracy on a loader.
20. **Task 20:** Showed 5 test images as Input, Ground truth, Prediction and Overlay (prediction in red).
21. **Task 21:** Filled in the results tables (above) and saved them as CSV files.

## Files

```text
notebook/assignment_resnet_unet.ipynb   the notebook with all outputs
results/                                graphs, confusion matrix, predictions and CSV tables
```

## How to Run
Open the notebook and run all cells from top to bottom.
