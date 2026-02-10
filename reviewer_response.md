# Response to Reviewers - Paper ID 165
## Deep Attention-based Sequential Ensemble Learning for BLE-Based Indoor Localization in Care Facilities

---

## REVIEWER #1

### Comment 1: Abstract - Mention ABC 2026 Challenge
> For readers not familiar in the challenge, it is recommended to mention the paper is written as part of the ABC 2026 Challenge in Abstract.

**Response:**
We have updated the Abstract to explicitly mention that this work is part of the ABC 2026 Activity and Location Recognition Challenge.

**Location of change:**
- **Abstract** (page 1): Added "developed as part of the ABC 2026 Activity and Location Recognition Challenge" in the second sentence.

**Revised text:**
> "To address this limitation, this paper, developed as part of the ABC 2026 Activity and Location Recognition Challenge, introduces Deep Attention-based Sequential Ensemble Learning (DASEL)..."

---

### Comment 2: Citations - Add challenge paper references
> Should cite challenge papers for source of data especially in using the map from the original papers in Fig.1. Following references to be cited:
> - [1] C. A. Garcia, Q. N. P. Vu, H. Kaneko, and S. Inoue, "A relabeling approach to signal patterns for beacon-based indoor localization in nursing care facility," International Journal of Activity and Behavior Computing, vol. 2024, no. 1, pp. 1–19, 2024.
> - [2] C. Garcia and S. Inoue, "Relabeling for indoor localization using stationary beacons in nursing care facilities," Sensors, vol. 24, no. 2, 2024.

**Response:**
We have added all three requested challenge paper citations to the bibliography and cited them appropriately throughout the manuscript.

**Location of changes:**

1. **Bibliography** (page 13-14): Added three new references:
   - **[b20]**: Garcia et al. 2024 IJABC paper (relabeling approach)
   - **[b21]**: Garcia & Inoue 2024 Sensors paper (relabeling for indoor localization)
   - **[b26]**: Garcia et al. 2026 "Decoding the Invisible" challenge summary paper

2. **Figure 1 caption** (Section 2.1, page 3): Added citation to source of floor map
   - Changed from: "5th floor map with beacon positions and room layout."
   - Changed to: "5th floor map with beacon positions and room layout \cite{b20, b26}."

3. **Dataset section** (Section 2.1, page 3): Added citation to challenge summary
   - Changed from: "\cite{b18}"
   - Changed to: "\cite{b18, b26}"

4. **Traditional ML section** (Section 2.3.1, page 5): Updated citation for Garcia relabeling method
   - Changed from: "\cite{b20}"
   - Changed to: "\cite{b20, b21}"

**Note:** References b21-b25 were renumbered to b22-b27 to accommodate the new citations. All in-text citations have been updated accordingly.

---

### Comment 3: Decode the Invisible Challenge citation
> Recommended for authors to mention the paper is part of the Decode the Invisible Challenge with the citation in reference:
> C. Garcia, U. Dobhal, L. Zhao, X. Min, N. Nahid, and S. Inoue. Decoding the Invisible: A Summary of the Location Recognition Challenge in Care Facilities. International Journal of Activity and Behavior Computing, 2026(2).

**Response:**
We have added this citation as requested.

**Location of changes:**
- **Bibliography** (page 14): Added as reference [b26]
- **Dataset section** (Section 2.1, page 3): Cited alongside [b18]
- **Figure 1 caption** (Section 2.1, page 3): Cited alongside [b20] as source of floor map

---

### Comment 4: Figure quality improvements
> Improve contrast or enlarge for better readability of Fig.2, 3, 4, 5, 7 and 8. Ensure all figure axes and legends are clearly readable.

**Response:**
We have improved the contrast and resolution of all requested figures. Specifically:

**Figures updated:**
- **Figure 2** (Class distribution): Enhanced contrast, enlarged axes labels and legend
- **Figure 3** (Traditional ML workflow): Increased resolution, improved text readability
- **Figure 4** (DASEL workflow): Enhanced contrast, enlarged component labels
- **Figure 5** (DASEL model architecture): Improved layer labels and connection clarity
- **Figure 7** (RSSI distribution comparison): Enlarged axes, enhanced legend readability
- **Figure 8** (Frequency distribution): Improved contrast, enlarged axes labels

All figure axes, legends, and annotations are now clearly readable at publication resolution.

---

### Comment 5: Proofreading
> Proofread the manuscript to correct grammatical errors and recheck sentence clarity.

**Response:**
We have carefully proofread the entire manuscript and corrected grammatical errors. Major improvements include:
- Clarified complex sentences in the Introduction and Discussion sections
- Fixed subject-verb agreement issues
- Improved clarity of technical descriptions in the Methodology section
- Enhanced readability of the Results and Analysis section

---

## REVIEWER #2

### Comment 1: Explicit research objectives
> Add explicit and measurable subsections or bullet lists of Goals for implemented methods, improved aspects, and evaluation targets according to the challenge.

**Response:**
We have completely restructured the final two paragraphs of the Introduction section to explicitly present our research objectives, improved aspects, and evaluation targets in a clear, numbered format.

**Location of change:**
- **Introduction section** (Section 1, page 2): Rewrote the last two paragraphs

**New structure includes:**

**Paragraph 1 - Research Objectives (three explicit goals):**
1. **Methodological goal**: Achieve breakthrough performance by leveraging temporal dependencies
2. **Technical improvements**: Address RSSI instability, handle sequence boundary challenges, manage class imbalance
3. **Evaluation targets**: Maximize macro F1 score with balanced performance across all room classes

**Paragraph 2 - Measurable Contributions:**
Lists five specific contributions with measurable achievements, including:
- The 0.2898 baseline performance (where traditional methods plateau)
- The 0.4438 DASEL performance (53.1% improvement)
- Explicit mention of ABC 2026 Challenge evaluation metric (macro F1 score)

**Key added content:**
> "This work addresses the location recognition component of the ABC 2026 Activity and Location Recognition Challenge, which evaluates methods using macro F1 score as the primary metric for location classification. Our research objectives are threefold: (1) Methodological goal: achieve breakthrough performance improvement by leveraging temporal dependencies...; (2) Technical improvements: address RSSI instability...; (3) Evaluation targets: maximize location classification macro F1 score..."

---

### Comment 2: Link attention mechanisms to data characteristics and performance
> The discussion has not fully linked attention mechanisms, data characteristics, and performance achievements.

**Response:**
We have substantially revised the Discussion section paragraph on bidirectional GRU architecture to explicitly link data characteristics → attention mechanism design → performance improvements.

**Location of change:**
- **Discussion section** (Section 4, page 10): Rewrote the paragraph beginning with "The bidirectional GRU architecture..."

**New content explicitly establishes:**

1. **Data characteristic**: Indoor localization data consists of temporal sequences with continuous human movement trajectories (not independent observations)

2. **Core architectural solution**: Bidirectional GRU captures sequential patterns → accounts for majority of performance improvement

3. **Data quality issue**: Not all timesteps are equally informative (stable periods vs. noisy transitions)

4. **Attention mechanism role**: Identifies and emphasizes key moments while downweighting noisy transitions → simplifies complex patterns

5. **Direct performance link**: This architecture-to-data matching → 53.1% performance improvement

**Key added content:**
> "The bidirectional GRU architecture with attention mechanism directly addresses the fundamental characteristic of indoor localization data: temporal sequences where human movements follow continuous trajectories rather than independent observations. The core breakthrough comes from the bidirectional GRU layers, which capture both past context (where the person came from) and future context (where they are going)—this sequential modeling alone accounts for the majority of performance improvement over traditional methods... This architectural design—matching the model structure to the inherent temporal dependencies and variable signal quality in real-world movement data—directly translates to the observed 53.1% performance improvement..."

---

## SUMMARY OF REVISIONS

### Citations and References:
- ✅ Added 3 new references (Garcia et al. papers)
- ✅ Updated Figure 1 caption to cite floor map source
- ✅ Updated all affected in-text citations (renumbered b21-b25 → b22-b27)
- ✅ All 27 references verified for correct sequential numbering and citation usage

### Content Improvements:
- ✅ Abstract explicitly mentions ABC 2026 Challenge participation
- ✅ Introduction restructured with explicit numbered research objectives
- ✅ Discussion enhanced to link attention mechanism → data characteristics → performance
- ✅ Manuscript proofread for grammatical errors and clarity

### Figure Quality:
- ✅ Enhanced contrast and resolution for Figures 2, 3, 4, 5, 7, and 8
- ✅ Enlarged axes labels and legends for readability
- ✅ All figures now meet publication quality standards

We believe these revisions have fully addressed all reviewer concerns and substantially improved the manuscript's clarity and completeness. We thank the reviewers for their valuable feedback.
