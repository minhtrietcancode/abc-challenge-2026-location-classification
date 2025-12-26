## Quick Summary
I'm working on the ABC 2026 location prediction challenge. After trying 7 different XGBoost approaches (all stuck at ~0.30 macro F1), I switched to LSTM sequential modeling and achieved ~0.48 macro F1 (~60% improvement). However, my current approach uses "ground truth segmentation" which is not realistic for production. I need help figuring out how to do inference on unlabeled continuous BLE data.

---

## Full Context

### The Challenge
- **Task**: Predict which room a person is in on the 5th floor of a care facility
- **Input Data**: Bluetooth Low Energy (BLE) beacon RSSI readings from 23 beacons
- **Evaluation Metric**: Macro F1 score (all rooms weighted equally)
- **Dataset**: 4 days of labeled data (~1.1M records after cleaning)
  - Day 1: ~600K records
  - Day 2: ~330K records
  - Day 3: ~145K records
  - Day 4: ~28K records

### What I've Tried (Approaches 1-7)
All using XGBoost with 1-second windowed aggregation:
1. **Baseline**: mean/std/count features → 0.28 F1
2. **Extended features**: added min/max → 0.30-0.31 F1
3. **Class weighting**: 3× minority weight → No improvement
4. **SMOTE**: Oversampling → No improvement
5. **Dominated beacon**: Added strongest beacon feature → No improvement
6. **Relabeling**: Temporal smoothing → No improvement (hurts macro F1)
7. **Zone classification**: Two-stage hierarchical → 0.30 F1 (error propagation)

**Plateau**: All XGBoost approaches stuck at ~0.30 macro F1

### Current Approach (Approach 8: LSTM Sequential)

**Key Innovation**: Model the temporal sequence of beacon appearances instead of treating each window independently.

**Hypothesis**: 
- RSSI signal strength values are noisy and unreliable
- Beacon appearance patterns (which beacons appear, how often) are more stable
- Temporal dependencies matter - the sequence of beacons over time is discriminative

**Current Pipeline**:
1. **Feature Engineering**: 
   - 23-dimensional beacon count vectors (one per timestamp)
   - Each value = percentage/count of readings from that beacon at that timestamp
   - Tested both percentage (normalized) and raw counts → nearly identical performance

2. **Sequence Creation** (THE "CHEATING" PART):
   ```python
   df['room_group'] = (df['room'] != df['room'].shift()).cumsum()
   ```
   - Uses ground truth room labels to identify continuous room visits
   - Each visit becomes one sequence (min length: 3, max length: 50)
   - This is "cheating" because we won't have room labels during inference

3. **LSTM Architecture**:
   ```
   Masking → LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3) 
   → Dense(32) → Dropout(0.2) → Dense(num_classes, softmax)
   ```

4. **Evaluation**: 4-fold cross-validation × 10 random seeds = 40 runs per approach

**Results**:
- **Approach 8a (Percentage features)**: 0.4792 ± 0.0890 macro F1
- **Approach 8b (Raw count features)**: 0.4804 ± 0.0793 macro F1
- **Improvement**: ~60% over XGBoost (0.48 vs 0.30)

**Key Findings**:
✅ Sequential modeling works - LSTM captures temporal patterns effectively
✅ Beacon appearance frequency matters more than RSSI values
✅ Percentage vs raw counts → nearly identical (use percentage for normalization)
✅ Macro F1 is granularity-agnostic (sequence-level ≈ frame-level predictions)

---

## THE PROBLEM I NEED HELP WITH

### Current Limitation
My LSTM approach uses **ground truth room labels** to segment sequences during both training and testing:
```python
df['room_group'] = (df['room'] != df['room'].shift()).cumsum()
```

This is a **controlled experiment** to validate that:
1. Sequential patterns exist in the data ✅
2. LSTM can learn these patterns ✅
3. The approach is fundamentally sound ✅

But it's **"cheating"** because during real inference on unlabeled test data:
- We don't know when room changes occur
- We can't use `room_group` to segment sequences
- We need an **automatic segmentation strategy**

### The Real Challenge
**How do I create sequences from unlabeled continuous BLE data during inference?**

### Unknown Competition Details (Organizers are amateur and didn't provide clear info)
❓ What format is the test data?
❓ What granularity of predictions is required (frame-level? sequence-level?)
❓ How will predictions be evaluated?
❓ What is the submission format?

I've confirmed through testing that **frame-level macro F1 ≈ sequence-level macro F1** (due to how macro averaging works), so granularity doesn't matter much for evaluation.

---

## WHAT I NEED FROM YOU

Please help me brainstorm and evaluate approaches for **production inference** that can work on unlabeled continuous BLE data:

### Possible Strategies I'm Considering

1. **Sliding Window**:
   - Fixed-length sequences (e.g., 50 timestamps)
   - Slide forward by N steps
   - Pros: Simple, guaranteed to work
   - Cons: May split room visits across windows

2. **Change Point Detection**:
   - Detect when beacon patterns shift significantly
   - Create sequences based on detected boundaries
   - Pros: Adaptive, mirrors training data structure
   - Cons: Complex, may miss subtle transitions

3. **Overlapping Windows with Voting**:
   - Multiple overlapping sequences
   - Majority vote or confidence weighting
   - Pros: Robust, can smooth predictions
   - Cons: Computationally expensive

4. **Continuous Sequence Prediction**:
   - Feed entire day as one long sequence
   - LSTM predicts at each timestep (may need architecture changes)
   - Pros: Most realistic, end-to-end
   - Cons: May require bidirectional LSTM, CRF, etc.

### What I Want You To Do

1. **Analyze the options above** - pros/cons, feasibility, expected performance
2. **Suggest additional approaches** I haven't considered
3. **Recommend which strategy to implement first** and why
4. **Help me think through the implementation details** for the chosen approach
5. **Consider potential issues** (edge cases, performance degradation, etc.)
6. **Search the web if needed** for:
   - Similar problems in location prediction / indoor positioning
   - LSTM inference strategies for sequence segmentation
   - Change point detection algorithms for time series
   - Best practices for deploying sequence models in production

### Important Constraints
- Must work on **unlabeled continuous BLE streams** (no ground truth boundaries)
- Should maintain performance as close to **0.48 macro F1** as possible
- Needs to be **implementable in Python** with TensorFlow/Keras
- Should be **computationally reasonable** (can't take forever to run)

---

## Files You Should Read

I'll provide you with two markdown files:
1. **README.md** - Full project documentation, data pipeline, problem context
2. **approach_note.md** - Detailed documentation of all 8 modeling approaches and results

Please read both carefully to understand:
- The data structure and preprocessing pipeline
- The 4-fold cross-validation setup
- Why previous approaches failed
- How the current LSTM approach works
- The "cheating" validation strategy and its purpose

---

## Expected Output

After reading the files and understanding the context, please provide:

1. **Your analysis** of the current situation
2. **Recommended inference strategy** (with clear reasoning)
3. **Step-by-step implementation plan** for the recommended approach
4. **Expected challenges** and how to address them
5. **Alternative approaches** to consider if the first one fails
6. **Research/resources** from web search if applicable

Let's work together to bridge the gap from this "cheating" validation (0.48 F1) to a production-ready system that works on unlabeled data!

---

## Additional Notes

- The competition organizers are amateur and haven't provided clear test data format or submission requirements
- I've tried contacting them but responses are slow/unclear
- I need to be ready for multiple scenarios (different test formats)
- Building a robust inference pipeline is now the top priority
- The model architecture and features are already validated - focus on **inference/deployment**

Please read the attached files and help me figure out the best inference strategy for production deployment.