# Roadmap for `sightseer`

This document acts as a sort of to-do list. It contains all the present requirements and features. The plan and order will be followed until the package's completion.

---

## The Components
`sightseer` is centered around 4 major components:

| Component | Description                                                               |
|-----------|---------------------------------------------------------------------------|
| Sightseer | Obtains image data or video footage                                       |
| Proc      | Provides image/frame-wise annotation and inter-format conversion tools    |
| Zoo       | Stores the wrappers over all state-of-the-art models and configs          |
| Serve     | Provides deployment and model serving protocols and services              |

---

## `v1.1.0` Planned Release

The upcoming version will tentatively contain the following features:

### Sightseer
- Video loading
- Webcam footage
- Screen recordings 

### Proc
- Inter-format data conversion for fine-tuning (XML/CSV/JSON/TFRecords)

---

## Build Schedule

### Sightseer

- [x] Webcam footage
- [x] Screen grab
- [x] Preloaded video footage

### Proc

- [x] `xml_to_csv`
- [x] `json_to_csv`
- [ ] `csv_to_tfrecord`
- [ ] `csv_to_xml`
- [ ] `csv_to_json`
- [ ] `tfrecord_to_csv`

### Zoo 

*Models*

- [x] YOLOv3 (`YOLOv3Client`)
- [ ] Mask RCNN (`MaskRCNNClient`)
- [ ] Fast RCNN (`FastRCNNClient`)
- [ ] Faster RCNN (`FASTERRCNNClient`)
- [ ] TinyYOLO (`TinyYOLOClient`)
- [ ] TensorFlow Object Detection (`TFODClient`)
- [ ] Single Shot Detector (`SSDClient`)
- [ ] TensorFlow Object Counting (`TFOCClient`)

*Add-ons*

- [ ] Finetuning framework
- [ ] Hardware acceleration support (CPU, GPU, TPU)
- [ ] Support for [TensorFlow 2.0](https://www.tensorflow.org/guide/effective_tf2)

### Serve

- [ ] Google Cloud Platform support
- [ ] Amazon Web Services support

---

## Contributions

If there are any changes to be made, please submit a PR or file an issue. Once reviewed and finalised, changes can be reflected in the `master` branch.