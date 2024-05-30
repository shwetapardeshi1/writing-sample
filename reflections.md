# Reflection on Article - Image Classification using CNNs

## Overview of Challenges and Process

My main goal for this article was to make the concepts of CNNs intuitive. I wanted readers to grasp what convolution does, why we use pooling, and how pooling filters important information. When I first learned about CNNs, I struggled with understanding why the feature map often has more dimensions than the input image and the reason for applying multiple kernels to the image. I know many people face these challenges, so I aimed to make these concepts clear.

The main issue I faced was the word limit. There are many underlying concepts to CNNs, such as the mathematics behind convolution, different layers like pooling, parameters like stride and padding, and various applications. To fit the article within the word limit, I decided not to include trivial information that can be easily accessed through a Google search.

The research process and writing process took more than 2 hours. Just like we do warmup before any exercise, I think we should read before writing anything (related/unrelated to the topic). I spent more than 3 hrs properly learning about the basics even though I have used CNNs such as ResNet before. Then the information filtering and structure of the article took a bit of my time. I also created SVG images I have used myself using draw.io tool.

I used ChatGPT to obtain the MNIST training code, but it turned out to be the exact code from PyTorch's blitz tutorial, so I cited the PyTorch documentation. Additionally, I used ChatGPT for Markdown syntax, attempting to create a toggle for the assessment answer, but was unsuccessful. I also tried using ChatGPT for shortening the text to meet the word limit; however, the responses did not sound very natural, so I ended up revising them myself.

## ChatGPT Conversation

**Question:** The output of torchvision datasets are PILImage images of range [0, 1] why do we transform them to Tensors of normalized range [-1, 1]?

**ChatGPT:**

The transformation of PILImage images from the range [0, 1] to Tensors of normalized range [-1, 1] in torchvision datasets is done for the following reasons:

- Improved Training Stability and Performance:
  Neural networks, including CNNs, often perform better when the input data is normalized. Normalizing the data helps in achieving faster convergence during training because it ensures that the network's weights are updated in a more consistent and stable manner.
- Consistent Scale Across Features:
  When the input values are within a similar range, it prevents any single feature from dominating the learning process due to its scale. By normalizing to the range [-1, 1], each pixel contributes equally, ensuring that the learning process is more balanced and effective.
- Compatibility with Activation Functions:
  Many activation functions, such as the hyperbolic tangent (tanh) function, work best when the input values are centered around zero. The tanh function, for instance, outputs values in the range [-1, 1]. When input values are normalized to this range, the activation function operates more efficiently, leading to better learning and gradient propagation.
- Preprocessing Consistency:
  Normalizing the input images to a standard range helps maintain consistency across different datasets and models. This consistency is crucial when applying transfer learning or fine-tuning pre-trained models, as these models are often trained on normalized data.
- Numerical Stability:
  Normalizing the input data helps in maintaining numerical stability during the computation of gradients. This stability is crucial for avoiding issues like vanishing or exploding gradients, which can hamper the training process.

By transforming the PILImage images to Tensors of normalized range [-1, 1], the model training process becomes more efficient, stable, and effective, leading to better performance and faster convergence.

**Follow Up Question:** But why -1, 1 and not 0, 1?

**ChatGPT:**
Normalizing to [-1, 1] instead of [0, 1] aligns better with activation functions like tanh, which operate more efficiently around zero, and it helps in achieving faster and more stable convergence during training by centering the data.

I changed the explanation from `Tanh` to `relu` since I was using `relu` in the code.

# Feedback

I am seeking feedback on the technical details included in the article. My main goal was to ensure that the amount of information provided on convolution and CNN architecture is sufficient. Additionally, I would like your opinion on the code section. Due to the word limit, I couldn't explain the code in much detail. In the assessment, what quantity and complexity level of questions do you anticipate? The questions I've included necessitate some brainstorming, as the answers are not explicitly stated in the article.
