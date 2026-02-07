# Skills Assessment

## Overview

**Congratulations on completing the Fundamentals of AI module!**

This module has covered a comprehensive range of theoretical concepts in **Artificial Intelligence**, **Machine Learning**, **Deep Learning**, and **Generative AI**. From foundational algorithms to state-of-the-art models, you've explored the core principles that power modern AI systems.

---

## Assessment Format

Given that this module was **entirely theoretical**, the skills assessment consists of **a few questions** designed to test your understanding of the theoretical content.

**Purpose:**
- ✅ Reinforce key concepts
- ✅ Verify comprehension of core algorithms
- ✅ Assess knowledge of fundamental architectures
- ✅ Prepare for practical applications in future modules

**Structure:**
- 5 questions covering various topics from the module
- Each question worth **+2 cubes**
- Total possible: **10 cubes**

---

## Questions

### Question 1 (2 cubes)

**Which probabilistic algorithm, based on Bayes' theorem, is commonly used for classification tasks such as spam filtering and sentiment analysis, and is known for its simplicity, efficiency, and good performance in real-world scenarios?**

---

#### Answer

**Naive Bayes**

---

#### Explanation

**Naive Bayes** is a probabilistic classification algorithm that applies **Bayes' theorem** with the "naive" assumption of **conditional independence** between features.

**Key characteristics:**

✅ **Based on Bayes' theorem:**
```
P(class | features) = P(features | class) × P(class) / P(features)
```

✅ **Naive assumption:**
- Features are conditionally independent given the class
- Simplifies computation
- Works surprisingly well despite this strong assumption

✅ **Common applications:**
- **Spam filtering**: Classify emails as spam or not spam
- **Sentiment analysis**: Determine sentiment (positive/negative) from text
- **Document classification**: Categorize documents by topic
- **Medical diagnosis**: Predict diseases based on symptoms

✅ **Advantages:**
- Simple and fast
- Works well with small datasets
- Handles high-dimensional data
- Good baseline model

**Real-world scenario:**

**Spam filtering:**
```
P(spam | "buy now", "limited offer") = 
    P("buy now", "limited offer" | spam) × P(spam) / P("buy now", "limited offer")
```

If this probability > 0.5, classify as spam.

**Why it's effective:**
- Fast training and prediction
- Requires small amount of training data
- Performs well even with the independence assumption
- Easy to update with new data

---

### Question 2 (2 cubes)

**What dimensionality reduction technique transforms high-dimensional data into a lower-dimensional representation while preserving as much original information as possible, and is widely used for feature extraction, data visualization, and noise reduction?**

---

#### Answer

**Principal Component Analysis (PCA)**

---

#### Explanation

**Principal Component Analysis (PCA)** is a dimensionality reduction technique that transforms high-dimensional data into a **lower-dimensional representation** while **preserving as much variance** (information) as possible.

**How it works:**

**1. Find principal components:**
- Directions of maximum variance in the data
- First PC: Direction of highest variance
- Second PC: Perpendicular to first, next highest variance
- And so on...

**2. Project data:**
- Transform original data onto principal components
- Keep only top k components
- Reduce from n dimensions to k dimensions

**Mathematical foundation:**
```
X_reduced = X · W_k

Where:
- X: Original data (n × d)
- W_k: Top k eigenvectors (d × k)
- X_reduced: Reduced data (n × k)
```

---

**Key applications:**

✅ **Feature extraction:**
- Reduce number of features
- Remove redundant information
- Improve model efficiency

✅ **Data visualization:**
- Reduce to 2D or 3D for plotting
- Visualize high-dimensional data
- Explore data structure

✅ **Noise reduction:**
- Remove low-variance components (often noise)
- Clean data
- Improve signal-to-noise ratio

✅ **Preprocessing:**
- Before training ML models
- Speed up training
- Prevent overfitting

---

**Example:**

**Image compression:**
```
Original image: 1000 × 1000 pixels = 1,000,000 dimensions
Apply PCA: Keep top 100 components
Compressed: 100 dimensions
Reconstruction: Approximately recover original image
Compression ratio: 10,000:1
```

**Why it's important:**
- Handles curse of dimensionality
- Reveals underlying structure
- Makes data more manageable
- Foundational technique in data science

---

### Question 3 (2 cubes)

**What model-free reinforcement learning algorithm learns an optimal policy by estimating the Q-value, which represents the expected cumulative reward an agent can obtain by taking a specific action in a given state and following the optimal policy afterward? This algorithm learns directly through trial and error, interacting with the environment and observing the outcomes.**

---

#### Answer

**Q-Learning**

---

#### Explanation

**Q-Learning** is a **model-free** reinforcement learning algorithm that learns an optimal policy by estimating the **Q-value** (action-value function).

**What is Q-value?**

**Definition:** Expected cumulative reward for taking action a in state s, then following the optimal policy.

```
Q(s, a) = Expected total reward from state s, taking action a, then acting optimally
```

---

**How Q-Learning works:**

**1. Initialize Q-table:**
```
Q(s, a) = 0 for all states s and actions a
```

**2. Interaction loop:**
```
For each episode:
    Start in state s
    
    While not terminal:
        1. Choose action a (ε-greedy)
        2. Take action a, observe reward r and next state s'
        3. Update Q-value:
           Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        4. s ← s'
```

**Update rule components:**
- **α**: Learning rate (how much to update)
- **γ**: Discount factor (importance of future rewards)
- **r**: Immediate reward
- **max Q(s',a')**: Best possible future value

---

**Key characteristics:**

✅ **Model-free:**
- No need to know environment dynamics
- Learns directly from experience
- No model of transitions or rewards

✅ **Off-policy:**
- Learns optimal policy (greedy)
- While following exploratory policy (ε-greedy)
- Can learn from any experience

✅ **Trial and error:**
- Agent interacts with environment
- Observes outcomes
- Updates estimates
- Gradually improves policy

✅ **Temporal Difference (TD) learning:**
- Updates based on difference between prediction and actual
- Learns from each step
- Don't need to wait for episode end

---

**Applications:**

**Game playing:**
- Learn to play games (Atari, board games)
- No game rules needed
- Discovers strategies through play

**Robotics:**
- Robot navigation
- Manipulation tasks
- Learns from interaction

**Resource management:**
- Traffic light control
- Energy management
- Network routing

---

**Example: Grid world navigation**

```
Agent starts at position (0,0)
Goal at position (4,4)

Through trial and error:
Episode 1: Random walk, finds goal eventually
Episode 2: Slightly better path
Episode 10: Efficient path emerging
Episode 100: Optimal path learned

Q-table learned:
Each state-action pair has estimated value
Policy: Always take action with highest Q-value
Result: Optimal path from any starting position
```

**Why it's fundamental:**
- Foundation of deep Q-learning (DQN)
- Simple yet powerful
- Widely applicable
- Proven convergence guarantees

---

### Question 4 (2 cubes)

**What is the fundamental computational unit in neural networks that receives inputs, processes them using weights and a bias, and applies an activation function to produce an output? Unlike the perceptron, which uses a step function for binary classification, this unit can use various activation functions such as the sigmoid, ReLU, and tanh.**

---

#### Answer

**Neuron** (also called an **artificial neuron** or **node**)

---

#### Explanation

A **neuron** is the **fundamental computational unit** in neural networks, inspired by biological neurons in the brain.

**Structure and operation:**

```
Inputs: x₁, x₂, ..., xₙ
         ↓
    [Weighted Sum]
    z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
         ↓
  [Activation Function]
       y = f(z)
         ↓
      Output: y
```

---

**Components:**

**1. Inputs (x₁, x₂, ..., xₙ):**
- Features or outputs from previous layer
- Can be numbers, pixel values, etc.

**2. Weights (w₁, w₂, ..., wₙ):**
- Learned parameters
- Control importance of each input
- Adjusted during training

**3. Bias (b):**
- Learned parameter
- Shifts activation function
- Allows flexibility

**4. Weighted sum (z):**
```
z = Σ(wᵢ × xᵢ) + b
```

**5. Activation function (f):**
- Introduces non-linearity
- Determines neuron output
- Various choices available

---

**Activation functions:**

Unlike the **perceptron** (which uses a step function for binary classification), neurons can use **various activation functions**:

**Sigmoid:**
```
f(z) = 1 / (1 + e⁻ᶻ)
Output: (0, 1)
Use: Binary classification, probabilities
```

**ReLU (Rectified Linear Unit):**
```
f(z) = max(0, z)
Output: [0, ∞)
Use: Hidden layers, most popular
```

**Tanh (Hyperbolic Tangent):**
```
f(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
Output: (-1, 1)
Use: Hidden layers, zero-centered
```

**Softmax (for output layer):**
```
f(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
Output: Probability distribution
Use: Multi-class classification
```

---

**Neuron vs Perceptron:**

| Feature | Perceptron | Neuron |
|---------|-----------|--------|
| **Activation** | Step function | Various (sigmoid, ReLU, tanh, etc.) |
| **Output** | Binary (0 or 1) | Continuous or binary |
| **Use case** | Simple binary classification | General-purpose, multi-class, regression |
| **Differentiable** | No | Yes (with smooth activation) |
| **Backpropagation** | Limited | Full support |
| **Expressiveness** | Linear only | Can capture non-linear patterns |

---

**Role in neural networks:**

**Building blocks:**
```
Neural Network = Multiple layers of neurons

Input Layer → Hidden Layer(s) → Output Layer
   ↓              ↓                  ↓
[Neurons]     [Neurons]          [Neurons]
```

**Information flow:**
1. Input neurons receive features
2. Hidden neurons extract patterns
3. Output neurons produce predictions

**Learning:**
- Weights and biases adjusted during training
- Backpropagation computes gradients
- Optimization algorithm updates parameters

---

**Why neurons are powerful:**

✅ **Non-linearity:**
- Activation functions enable non-linear mappings
- Can learn complex patterns
- Multiple neurons combine for expressiveness

✅ **Composition:**
- Layers of neurons create deep networks
- Hierarchical feature learning
- Universal approximation capability

✅ **Differentiability:**
- Smooth activation functions are differentiable
- Enable gradient-based learning
- Support backpropagation

✅ **Parallelism:**
- Neurons can be computed in parallel
- Efficient on GPUs
- Scalable to large networks

---

**Example: Image classification**

```
Input neuron: Receives pixel value (e.g., 0.8)
Weight: 0.5
Bias: -0.2

Weighted sum: z = 0.5 × 0.8 + (-0.2) = 0.2
Activation (ReLU): f(0.2) = 0.2
Output: 0.2

This output becomes input to next layer neurons.

Through many layers:
Early layers: Detect edges, textures
Middle layers: Detect shapes, parts
Late layers: Detect objects
Output layer: Class probabilities (cat: 0.9, dog: 0.1)
```

**The neuron is the fundamental unit** that, when combined in networks, enables the powerful capabilities of modern deep learning systems!

---

### Question 5 (2 cubes)

**What deep learning architecture, known for its ability to process sequential data like text by capturing long-range dependencies between words through self-attention, forms the basis of large language models (LLMs) that can perform tasks such as translation, summarization, question answering, and creative writing?**

---

#### Answer

**Transformer**

---

#### Explanation

The **Transformer** is a deep learning architecture introduced in the 2017 paper *"Attention Is All You Need"* by Vaswani et al. It revolutionized natural language processing and forms the **basis of modern large language models (LLMs)**.

**Why Transformers are revolutionary:**

✅ **Self-attention mechanism:**
- Each word attends to all other words
- Captures relationships regardless of distance
- Parallel processing (vs sequential in RNNs)

✅ **Long-range dependencies:**
- Can relate words far apart in text
- No vanishing gradient problem
- Understands context across entire sequence

✅ **Scalability:**
- Parallelizable architecture
- Efficient on modern hardware (GPUs/TPUs)
- Can scale to billions of parameters

---

**Key architecture components:**

**1. Self-Attention:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k)·V

Where:
- Q (Query): What we're looking for
- K (Key): What each word offers
- V (Value): Actual information
```

**2. Multi-Head Attention:**
- Multiple attention mechanisms in parallel
- Each head learns different relationships
- Captures syntax, semantics, entities, etc.

**3. Positional Encoding:**
- Adds position information (since no recurrence)
- Sine/cosine functions
- Preserves word order

**4. Feed-Forward Networks:**
- Process each position
- Non-linear transformations
- Consistent across positions

**5. Layer Normalization:**
- Stabilizes training
- Faster convergence

**6. Residual Connections:**
- Skip connections around each sub-layer
- Helps gradient flow
- Enables very deep networks

---

**Transformer architecture:**

```
Input Text
    ↓
[Token Embedding + Positional Encoding]
    ↓
┌─────────────────────────────┐
│      Encoder Stack          │
│  ┌───────────────────────┐  │
│  │ Multi-Head Attention  │  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  Feed-Forward Network │  │
│  └───────────────────────┘  │
│  (Repeated N times)         │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│      Decoder Stack          │
│  ┌───────────────────────┐  │
│  │ Masked Self-Attention │  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  Cross-Attention      │  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  Feed-Forward Network │  │
│  └───────────────────────┘  │
│  (Repeated N times)         │
└─────────────────────────────┘
    ↓
[Output Probabilities]
```

---

**Variants:**

**Encoder-only (BERT-style):**
- Bidirectional context
- Good for understanding tasks
- Applications: Classification, NER, QA

**Decoder-only (GPT-style):**
- Autoregressive generation
- Good for generation tasks
- Applications: Text generation, completion

**Encoder-Decoder (Original Transformer):**
- Full architecture
- Good for seq-to-seq tasks
- Applications: Translation, summarization

---

**Large Language Models built on Transformers:**

**GPT series (OpenAI):**
- GPT-1: 117M parameters
- GPT-2: 1.5B parameters
- GPT-3: 175B parameters
- GPT-4: ~1.76T parameters
- Decoder-only architecture

**BERT (Google):**
- 110M - 340M parameters
- Encoder-only
- Bidirectional understanding

**T5 (Google):**
- Text-to-text framework
- Encoder-decoder
- Up to 11B parameters

**LLaMA (Meta):**
- Open-source
- 7B to 70B parameters
- Efficient decoder-only

**Claude (Anthropic):**
- Large context window
- Constitutional AI
- Based on Transformer architecture

---

**Capabilities enabled by Transformers:**

✅ **Machine Translation:**
```
English: "How are you?"
→ Transformer →
French: "Comment allez-vous?"
```

✅ **Text Summarization:**
```
Long article (1000 words)
→ Transformer →
Summary (100 words)
```

✅ **Question Answering:**
```
Context: [Document about Einstein]
Question: "When was Einstein born?"
→ Transformer →
Answer: "1879"
```

✅ **Creative Writing:**
```
Prompt: "Write a story about a robot"
→ Transformer →
Generated story: [Creative narrative]
```

✅ **Code Generation:**
```
Comment: "# Function to sort a list"
→ Transformer →
Code: [Complete Python function]
```

---

**Why Transformers are the foundation of modern NLP:**

✅ **Performance:**
- State-of-the-art on virtually all NLP benchmarks
- Consistent improvements with scale
- Generalizes well

✅ **Flexibility:**
- Handles variable-length sequences
- Works for many tasks
- Easy to adapt

✅ **Efficiency:**
- Parallelizable training
- Faster than RNNs
- Scales to massive datasets

✅ **Transfer learning:**
- Pre-train on massive data
- Fine-tune for specific tasks
- Few-shot and zero-shot learning

---

**Example: Self-Attention in action**

**Sentence:** "The cat sat on the mat because it was soft."

**Self-attention helps the model understand:**
- "it" refers to "mat" (not "cat")
- "sat" is the action
- "soft" describes "mat"
- "because" indicates causality

**Attention scores:**
```
"it" → "mat": 0.92 (high attention)
"it" → "cat": 0.05 (low attention)
"soft" → "mat": 0.88 (high attention)
"sat" → "cat": 0.85 (high attention)
```

This allows the model to correctly interpret the sentence and generate appropriate continuations or answers.

---

**Impact on AI:**

The Transformer architecture has fundamentally changed AI:
- Enabled ChatGPT, GPT-4, Claude, Gemini
- Powers modern search engines
- Basis for text-to-image models (DALL-E 2, Stable Diffusion)
- Extended to vision (Vision Transformers), audio, video, proteins
- Continues to drive AI progress

**The Transformer is arguably the most important architecture in modern AI**, forming the foundation of virtually all state-of-the-art language models and many other AI systems!

---

## Assessment Complete! 🎉

**Congratulations!** You've demonstrated understanding of the core concepts in the **Fundamentals of AI** module:

✅ **Machine Learning algorithms**: Naive Bayes, PCA
✅ **Reinforcement Learning**: Q-Learning
✅ **Neural Networks**: Neurons as fundamental units
✅ **Modern AI architectures**: Transformers and LLMs

**Total score:** 10 cubes (5 questions × 2 cubes)

---

## What's Next?

With the **Fundamentals of AI** module complete, you're ready to move on to **Module 02** and beyond, where you'll apply these theoretical foundations to:

**AI Security topics:**
- Prompt Injection attacks
- Jailbreaking techniques
- Model Inversion
- Adversarial examples
- Data poisoning
- Model extraction
- Privacy attacks
- And much more!

**The theoretical knowledge you've gained here will be essential** for understanding the vulnerabilities, attack vectors, and defense mechanisms in AI systems.

**Great work completing Module 01!** 🚀

---

## Key Takeaways

**From this module, remember:**

1. **Machine Learning**: Supervised, unsupervised, reinforcement learning paradigms
2. **Classical ML**: Linear/logistic regression, decision trees, SVMs, k-NN, clustering
3. **Deep Learning**: Neural networks, CNNs, RNNs, LSTMs, attention mechanisms
4. **Generative AI**: GANs, VAEs, diffusion models, autoregressive models
5. **Modern LLMs**: Transformers, self-attention, tokenization, fine-tuning
6. **Fundamentals**: Loss functions, optimization, regularization, evaluation metrics

**These concepts are the foundation** for understanding AI red teaming, adversarial machine learning, and AI security in the modules ahead!
