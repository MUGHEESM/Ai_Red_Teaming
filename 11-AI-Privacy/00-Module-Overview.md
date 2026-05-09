# Module Overview: AI Privacy

## Module Details

**Category:** Defensive  
**Last Updated:** 5 months ago  
**Module Progress:** 4.76% completed  
**Sections:** 21 sections  
**Interactive Exercises:** 3  
**Assessments:** 0  
**Badge of Completion:** 20 Cubes

## Module Description

This module explores privacy attacks against machine learning models and the differential privacy defenses that protect models from such attacks.

## Module Summary

Machine learning models are trained to learn patterns, not memorize individual bits of data. Yet every model trained on personal data carries a hidden risk: it can reveal which specific individuals were in its training set. This privacy violation, known as membership inference, represents one of the core threats to ML systems deployed on sensitive data.

The attack exploits a fundamental tension in machine learning: models tend to behave differently on data they have seen during training versus data they have not. This behavioral gap, caused by overfitting, creates a detectable membership fingerprint. An attacker with access only to model predictions can determine whether a particular person's data was used for training. If the model was trained exclusively on cancer patients, successfully identifying someone as a training member reveals their medical status.

This module provides an exploration of both attack and defense:

- Membership Inference Attacks (MIA) using the shadow model methodology introduced by Shokri et al. to train attack classifiers that detect membership based on prediction confidence patterns.
- Understanding differential privacy
- DP-SGD (Differentially Private Stochastic Gradient Descent), which modifies training through per-sample gradient clipping and calibrated noise injection to limit any individual's influence on model parameters.
- PATE (Private Aggregation of Teacher Ensembles), which achieves privacy through architectural separation, training multiple teachers on disjoint data partitions and using noisy vote aggregation to label public data for student training.

This module is broken into sections with hands-on exercises for implementing attacks and defenses. The DP-SGD section concludes with a skills assessment requiring submission to a validation server. It concludes with a practical skills assessment to validate your understanding.

You can start and stop at any time and resume where you left off. There is no time limit or grading, but you must complete all exercises and the skills assessment to receive the maximum cubes and have the module marked as complete in any selected paths.

To ensure a smooth learning experience, the following skills are mandatory: solid Python proficiency, familiarity with PyTorch, and understanding of neural network training, optimization, and evaluation metrics.

## Recommended Before Starting

A firm grasp of the following modules is recommended before starting:

- Fundamentals of AI
- Applications of AI in InfoSec
- Introduction to Red Teaming AI
- Prompt Injection Attacks
- AI Data Attacks
- AI Evasion Foundations
- AI Evasion - First-Order Attacks
- AI Evasion - Sparsity Attacks

Pwnbox will not be a good experience for this module. It is HIGHLY recommended to use your own PC/Laptop for the practicals.

## Creators

- PandaSt0rm

## Sections

1. Introduction
2. Shadow Model Attack
3. DP-SGD
4. Private Aggregation of Teacher Ensembles
5. Skills Assessment

## Module Progress

4.76% completed

