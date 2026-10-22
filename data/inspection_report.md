# Dataset Inspection Report (Phase 0)

## 1. File Metadata
- **File Name:** `logbooks.csv`
- **File Path:** `C:\Users\r1tpr\.gemini\antigravity\scratch\maintenance-copilot\data\raw\logbooks.csv`
- **File Size:** 0.48 MB (504,992 bytes)
- **Detected Encoding:** `utf-8-sig`
- **Byte Order Mark (BOM):** `Present (UTF-8-SIG required)`
- **Total Records:** 6,169
- **Total Fields/Columns:** 3

## 2. Schema and Field Breakdown
| Field | Raw Name | Missing / Null | Missing % | Unique Values | Primary Key Candidate? | Avg Length | Min Length | Max Length |
|---|---|---|---|---|---|---|---|---|
| `IDENT` | `'IDENT'` | 0 | 0.00% | 6,169 (100%) | **YES (100% Unique)** | 6.0 | 6 | 6 |
| `PROBLEM` | `'PROBLEM'` | 0 | 0.00% | 3,591 | NO | 36.2 | 9 | 60 |
| `ACTION` | `'ACTION'` | 0 | 0.00% | 2,878 | NO | 35.0 | 2 | 55 |

## 3. Data Integrity & Key Findings
- **Primary Key (`IDENT`):** 100% unique across all 6,169 records. Numerical sequential identifier (starts at `100001`, ends at `106169`). Zero missing values.
- **Discrepancy Text (`PROBLEM`):** Aviation discrepancy narrative describing the reported defect. Zero missing values. Average length: 36.2 characters (max: 60 characters).
- **Corrective Action Text (`ACTION`):** Maintenance corrective action narrative. Zero missing values. Average length: 35.0 characters (max: 55 characters).

## 4. Discovered Data Quality & Preprocessing Hazards
1. **UTF-8 Byte Order Mark (BOM):** The CSV file begins with `\xef\xbb\xbf`, meaning standard `utf-8` decoding yields `'\ufeffIDENT'`. Ingestion must strictly use `utf-8-sig` to prevent dictionary key mismatches.
2. **All-Caps Noise & Truncations:** Problem and Action descriptions are written entirely in UPPERCASE with frequent domain-specific abbreviations (`CYL`, `A/C`, `IAW`, `L/H`, `R/H`, `ENG`, `MAG`, `CHT`, `EGT`, `OPS`, `CK'ED`, `W/IN`, `O/B`, `I/B`).
3. **Incomplete Maintenance Manual References:** Several corrective action entries end with trailing whitespace after `IAW` (In Accordance With) where a technical manual reference was redacted or omitted (e.g., `FABRICATED PATCH OF LIKE MATERIAL & RIVETED IAW `).
4. **No Destructive Overwrite Required:** The original CSV is completely clean of null rows, delimiter corruptions, or ID collisions.

## 5. Sample Records
```json
[
  {
    "IDENT": "100001",
    "PROBLEM": "ENGINE IDLE OVERRIDE KILLED ENGINE.",
    "ACTION": "TRIED TO ADJUST IDLE SEVERAL TIMES, WOULDN'T ADJUST."
  },
  {
    "IDENT": "100002",
    "PROBLEM": "ENGINE IDLE OVERRIDE KILLED ENGINE.",
    "ACTION": "REMOVED & REPLACED FUEL SERVO"
  },
  {
    "IDENT": "100003",
    "PROBLEM": "ENGINE IDLE OVERRIDE KILLED ENGINE.",
    "ACTION": "A/C WAS RUN UP, SET IDLE SPEED, MIXTURE OK, NO LEAKS NO"
  }
]
```

## 6. Exit Criteria Status
- [x] Dataset successfully read without data loss or corruption.
- [x] Actual schema and record count verified (6,169 records, 3 columns: `IDENT`, `PROBLEM`, `ACTION`).
- [x] Primary key identifier validated (`IDENT` is guaranteed 100% unique).
- [x] Data-quality issues documented (UTF-8 BOM, all-caps noisy jargon, trailing references).
- [x] Original raw dataset preserved immutable at `data/raw/logbooks.csv`.