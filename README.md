# ITI123-Generative AI And Deep Learning Project

Main Data Set
- https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k/data
- https://odir2019.grand-challenge.org/dataset/

From the dataset kaggle page:

> Ocular Disease Intelligent Recognition (ODIR) is a structured ophthalmic database of 5,000 patients with age, 
> color fundus photographs from left and right eyes and doctors' diagnostic keywords from doctors.

> This dataset is meant to represent ‘‘real-life’’ set of patient information collected by Shanggong Medical Technology Co., Ltd. from different hospitals/medical centers in China. In these institutions, fundus images are captured by various cameras in the market, such as Canon, Zeiss and Kowa, resulting into varied image resolutions. Annotations were labeled by trained human readers with quality control management. They classify patient into eight labels including:

> Normal (N),
> Diabetes (D),
> Glaucoma (G),
> Cataract (C),
> Age related Macular Degeneration (A),
> Hypertension (H),
> Pathological Myopia (M),
> Other diseases/abnormalities (O)

Also from the ODIR-2019 grand challenge website:

> The 5,000 patients in this challenge are divided into training, off-site testing and on-site testing subsets. Almost 4,000 cases are used in training stage while  others are for testing stages (off-site and on-site). Table 2 shows the distribution of case number with respect to eight labels in different stages. **Note: one patient may contains one or multiple labels.**

Some data set process the ODIR-2019 as multi-label dataset but for  https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k/data, this is annotated as multi-class dataset. For simplicity and also due to time constraint, we tackle this as multi-class dataset instead of multi-label.

To do multi-label dataset, we need to manually process keywords for each image and detect more than one categories. Example of this is done in :
- https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning
- https://github.com/madhava20217/Ocular-Disease-Multiclass-Identification-ODIR19-

References:
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9230753/ Multi-Label Fundus Image Classification Using Attention Mechanisms and Feature Fusion  Zhenwei Li 1,*, Mengying Xu 1, Xiaoli Yang 1, Yanqi Han 1
- https://www.kaggle.com/datasets/efiyearcan/odir-5k-preprocessed-with-clahe
- https://www.kaggle.com/code/ahmetselukkren/preprocessing-with-clahe
- https://www.kaggle.com/code/umangtri/fundus-project-experiments
- https://www.kaggle.com/datasets/rohitrawat25/combined-fundus-images

Extra Data Set that can be incorporated in next stage
- Bangladesh Eye Hospital Dataset for Glaucoma:
    - https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/BEH.md
    - https://www.kaggle.com/datasets/kabirjaan/beh-glaucoma
- Hypertension
    - https://www.kaggle.com/datasets/harshwardhanfartale/hypertension-and-hypertensive-retinopathy-dataset