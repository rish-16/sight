# Roadmap for `sight`

This document acts as a sort of to-do list. It contains all the present requirements and features. The plan and order will be followed until the package's completion.

---

## The Components
`sight` is centered around 4 major components:

| Component | Description                                                      |
|-----------|------------------------------------------------------------------|
| Sightseer | Obtains image data or video footage                              |
| Proc      | Provides image/frame-wise annotation tools                       |
| Zoo       | Stores the wrappers over all state-of-the-art models and configs |
| Serve     | Provides deployment and model serving protocols and services     |

---

## Build Schedule

### Sightseer

- [x] Webcam footage
- [x] Screen grab
- [x] Preloaded video footage
- [ ] Realtime streaming

### Proc

- [x] `xml_to_csv`
- [x] `json_to_csv`
- [ ] `csv_to_tfrecord`
- [ ] `csv_to_xml`
- [ ] `csv_to_json`
- [ ] `tfrecord_to_csv`

### Zoo 

**Models**

- [ ] YOLO9000
- [ ] Mask RCNN
- [ ] Fast/Faster RCNN
- [ ] TensorFlow Object Detection
- [ ] Single Shot Detector

**Add-ons** 

- [ ] Finetuning framework
- [ ] Multi-hardware support (CPU, GPU, TPU)

### Serve

- [ ] Google Cloud Platform support
- [ ] Amazon Web Services support
- [ ] PySpark for realtime streaming

---

## Contributions

If there are any changes to be made, please submit a PR or file an issue. Once reviewed and finalised, changes can be reflected in the `master` branch.