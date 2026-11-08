# dataset_v1 readiness report

This is a PREVIEW split only. No frozen dataset was created.

## Rows by split, source, generator, verification, and language

```text
split  source        generator       human_verified  language
test   ai_generated  batch01_claude  True            en           12
                                                     hi           14
                                                     hinglish     14
                     batch02_claude  True            en           12
                                                     hi           12
                                                     hinglish     16
                     batch03_claude  True            en           12
                                                     hi           14
                                                     hinglish      5
                     batch04_claude  True            en           13
                                                     hi           10
                     batch05_claude  True            en           10
                                                     hi            1
                     batch06_claude  True            en            5
train  ai_generated  chatgpt_train   False           en          160
                                                     hi          160
                                                     hinglish    205
val    ai_generated  batch03_claude  True            hinglish      9
                     batch04_claude  True            hi            2
                                                     hinglish     10
                     batch05_claude  True            hi           13
                                                     hinglish     15
                     batch06_claude  True            en            7
                                                     hi           13
                                                     hinglish     14
                     batch07_claude  True            en           13
                                                     hi           13
                                                     hinglish     13
```

## Rows by split, urgency, and situation label

### Urgency

```text
split  urgency         
test   immediate_danger     21
       info_seeking         25
       non_urgent           44
       urgent               60
train  immediate_danger     56
       info_seeking        103
       non_urgent          159
       urgent              207
val    immediate_danger      5
       info_seeking         47
       non_urgent           44
       urgent               26
```

### Situation labels

```text
                      train  val  test
street_harassment        63   17    21
stalking                 92   13    23
domestic_violence       111   24    37
workplace_harassment     78   22    21
dowry_harassment         63   16    18
cyber_abuse             119   23    28
sexual_violence          45    7    19
other                    54   18    20
```

## Allocation share

- PASS — test verified-evaluation share: 150 (required 150)

## Test urgency counts

- PASS — test immediate_danger: 21 (required 5)
- PASS — test urgent: 60 (required 5)
- PASS — test non_urgent: 44 (required 5)
- PASS — test info_seeking: 25 (required 5)

## Test language counts

- PASS — test hi: 51 (required 30)
- PASS — test hinglish: 35 (required 30)
- PASS — test en: 64 (required 30)

## Validation language counts

- PASS — val hi: 41 (required 20)
- PASS — val hinglish: 61 (required 20)
- PASS — val en: 20 (required 20)

## Validation immediate-danger count

- PASS — val immediate_danger: 5 (required 5)

## Test situation-label counts

- PASS — test street_harassment: 21 (required 3)
- PASS — test stalking: 23 (required 3)
- PASS — test domestic_violence: 37 (required 3)
- PASS — test workplace_harassment: 21 (required 3)
- PASS — test dowry_harassment: 18 (required 3)
- PASS — test cyber_abuse: 28 (required 3)
- PASS — test sexual_violence: 19 (required 3)
- PASS — test other: 20 (required 3)

## Training situation-label counts

- PASS — train street_harassment: 63 (required 30)
- PASS — train stalking: 92 (required 30)
- PASS — train domestic_violence: 111 (required 30)
- PASS — train workplace_harassment: 78 (required 30)
- PASS — train dowry_harassment: 63 (required 30)
- PASS — train cyber_abuse: 119 (required 30)
- PASS — train sexual_violence: 45 (required 30)
- PASS — train other: 54 (required 30)

## Freeze rules

- PASS — total rows: 797 (required 400)
- PASS — human_verified evaluation rows: 272 (required 150)

## Generator isolation

- Train generators: chatgpt_train
- Val/test generators: batch01_claude, batch02_claude, batch03_claude, batch04_claude, batch05_claude, batch06_claude, batch07_claude
- PASS — no generator overlap.

## Exact freeze command (not run)

```powershell
python -m sakhi.src.data sakhi/data/raw/dataset_v1.csv --freeze
```
