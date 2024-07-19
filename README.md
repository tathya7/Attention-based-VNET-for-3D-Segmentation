# VNET with Attention Mechanism for Segmentation of 3D MRI and CT Aorta Dissection Images

The segmentation of 3D medical images, especially for complex structures like the aorta, presents significant challenges due to the varying shapes, sizes, and textures of anatomical features. Accurate segmentation is crucial for diagnosis, treatment planning, and disease monitoring. This project focuses on leveraging a VNET architecture with an attention mechanism to enhance the segmentation accuracy of 3D MRI and CT images, specifically targeting Type-B Aortic Dissection (TBAD).

## Dataset

A 3D Computed Tomography (CT) image dataset, ImageTBAD, for segmentation of Type-B Aortic Dissection is published. ImageTBAD contains 100 3D CT images, which is of decent size compared with existing medical imaging datasets.

ImageTBAD contains a total of 100 3D CTA images gathered from Guangdong Peoples' Hospital Data from January 1, 2013, to April 23, 2019. Images are acquired from a variety of scanners (GE Medical Systems, Siemens, Philips), resulting in large variance in voxel size, resolution, and imaging quality. All the images are pre-operative TBAD CTA images whose top and bottom are around the neck and the brachiocephalic vessels, respectively, in the axial view. The segmentation labeling is performed by a team of two cardiovascular radiologists who have extensive experience with TBAD. The segmentation labeling of each patient is fulfilled by one radiologist and checked by the other. The segmentation includes three substructures: TL, FL, and FLT. There are 68 images containing FLT while 32 images are free of FLT.

[Dataset on Kaggle](https://www.kaggle.com/datasets/xiaoweixumedicalai/imagetbad)

[Dataset Reference](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.732711/full)

## Model Architecture

The VNET with attention mechanism is designed for the segmentation of 3D MRI and CT Aorta Dissection Images. The architecture leverages the strengths of the V-Net model and incorporates attention mechanisms to improve segmentation accuracy.

### V-Net

V-Net is a fully convolutional neural network that is specifically designed for volumetric (3D) medical image segmentation. It consists of:

- **Encoder Path:** This path comprises a series of convolutional layers, each followed by a rectified linear unit (ReLU) and a downsampling layer (either strided convolution or max pooling). This path captures the context by progressively reducing the spatial dimensions while increasing the feature dimensions.
- **Decoder Path:** This path consists of convolutional layers followed by upsampling layers (transposed convolutions) that gradually restore the spatial dimensions while reducing the feature dimensions. Skip connections from the encoder path to the decoder path are used to retain high-resolution features.

### Attention Mechanism

The attention mechanism is integrated into the V-Net architecture to enhance the segmentation performance by focusing on the most relevant parts of the input image. The attention mechanism typically involves:

- **Attention Gates:** These gates are used in the skip connections of the U-Net architecture. They allow the network to weigh the importance of features coming from different scales.
- **Self-Attention:** This mechanism allows the network to capture long-range dependencies by computing attention scores for each voxel with respect to every other voxel in the feature map.

### Detailed Layers

1. **Input Layer:** 3D volume input.
2. **Encoder Block:** Multiple layers of 3D convolutions, each followed by ReLU activation and downsampling.
3. **Attention Gates:** Placed at skip connections to refine the features passed to the decoder.
4. **Decoder Block:** Multiple layers of 3D transposed convolutions, each followed by ReLU activation and upsampling.
5. **Output Layer:** Final 3D convolution layer with a sigmoid or softmax activation function for segmentation map output.

### Summary

- **Input:** 3D MRI/CT images.
- **Output:** Segmented 3D volumes indicating TL, FL, and FLT regions.
- **Loss Function:** Dice coefficient loss or a combination of Dice and cross-entropy loss.
- **Optimizer:** Adam optimizer or any suitable gradient-based optimizer.

## Results

<!-- Add your results here -->
![Result](https://github.com/Iaryan-21/VNET_AMC/blob/main/TBAD-108.png)

## References

1. [https://doi.org/10.1007/978-3-031-43901-8_63](https://doi.org/10.1007/978-3-031-43901-8_63)
2. [https://arxiv.org/abs/2007.10732](https://arxiv.org/abs/2007.10732)
3. [https://link.springer.com/chapter/10.1007/978-3-031-16452-1_46](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_46)
