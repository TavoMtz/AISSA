# AISSA: Physical AI Assistant for Office Visits Management

**AISSA** is a physical AI assistant designed to streamline and automate reception, check-in, and visitor management in office environments. By combining computer vision, natural language processing, and dedicated hardware, AISSA manages initial visitor interactions smoothly, efficiently, and securely.

---

## Project Overview

The system operates under an end-to-end systems engineering framework that bridges the physical and digital worlds:

1. **Perception:** The assistant detects the presence of a human in the physical space using computer vision modules.
2. **Interaction (Voice & Brain):** Once a visitor is detected, a conversational voice interface activates to capture speech, understand intent via a Large Language Model (LLM), and deliver natural spoken responses.
3. **Management:** The system coordinates office access or routes the visitor based on their needs and the established workplace protocols.

---

## Repository Structure

The repository is organized following standard project management and systems engineering methodologies:

```text
├── Administrative/
│   ├── Project Charter
│   ├── Project Management Plan
│   └── HW Requirements
├── Systems Engineering/
│   ├── Requirements Document (FR & NFR)
│   └── SysML Architecture Document (2_2_D_Project_T)
└── Source Code (Cycle 1)/
    ├── Perception/
    │   ├── human_detection.py
    │   ├── test_live.py
    │   └── test_inferece.py
    └── Voice_LLM/
        └── interaction/
            ├── main.py
            ├── config.py
            ├── ears.py
            ├── aissaBrain.py
            └── mouth.py
```
## Authors
Gustavo Martinez, Miguel Hernandez, Lucio Ruiz, Rodrigo Bernal and Enrique Gracian
