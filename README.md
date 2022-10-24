# Flash Attention
This document was prepared **only** for DS-5899 Paper Presentation at Vanderbilt University.

# Motivation of Flash Attention: Modeling Longer Sequences
* **NLP**: Large context required to understand books, plays, and instruction manuals
* **Computer Vision**: Higher resolution can lead to a better and more robust insight
* **Time series, audio, video, medical imaging**: Data are intrinsicly modeled as sequences of multiple steps

# Attention is the HEART of Transformers
<img src="https://user-images.githubusercontent.com/25111091/197630239-df4a88d6-7bd6-4d81-88cd-f3beae23fb9e.png" width="500">

# Background: Attention is Bottlenecked by Memory Reads/Writes
![Screen Shot 2022-10-24 at 4 12 20 PM](https://user-images.githubusercontent.com/25111091/197630379-74042ca2-a8f1-4c29-b029-c4e7019a79f7.png)
**O** = Dropout(Softmax(Mask(**QK** <sup>**T**</sup>)))**V**
## Question 1
Is Attention **Computer-Bound** or **Memory-Bound**?

# Flash Attention
![Screen Shot 2022-10-24 at 4 31 49 PM](https://user-images.githubusercontent.com/25111091/197633649-e77905cc-2025-4900-b003-28e0bc7be93a.png)
![Screen Shot 2022-10-24 at 4 32 01 PM](https://user-images.githubusercontent.com/25111091/197633680-7bca7826-287e-44ba-b97e-d8eee189289b.png)

# Standard Attention Implementation
![Screen Shot 2022-10-24 at 4 16 04 PM](https://user-images.githubusercontent.com/25111091/197633381-886b30ad-027d-4fde-8862-260fc79d477d.png)

# Flash Attention Implementation
![Screen Shot 2022-10-24 at 4 13 45 PM](https://user-images.githubusercontent.com/25111091/197630869-d6a48fba-d4f1-4027-ae48-7fc3a4a820ad.png)
