// Fallback metrics in case /metrics.json cannot be fetched over the network
export const fallbackMetrics = {
  project: "GuardianEye",
  version: "0.1.0",
  evaluated_at: "2026-10-26T20:00:00Z",
  status: "FALLBACK_SAMPLE_DATA",
  training_history: [
    { epoch: 1, image_train_loss: 1.342, image_val_loss: 1.182, image_train_acc: 0.442, image_val_acc: 0.521, audio_train_loss: 0.684, audio_val_loss: 0.612, audio_train_acc: 0.584, audio_val_acc: 0.672 },
    { epoch: 2, image_train_loss: 1.121, image_val_loss: 0.985, image_train_acc: 0.571, image_val_acc: 0.638, audio_train_loss: 0.572, audio_val_loss: 0.521, audio_train_acc: 0.702, audio_val_acc: 0.751 },
    { epoch: 3, image_train_loss: 0.964, image_val_loss: 0.864, image_train_acc: 0.648, image_val_acc: 0.704, audio_train_loss: 0.495, audio_val_loss: 0.458, audio_train_acc: 0.771, audio_val_acc: 0.812 },
    { epoch: 4, image_train_loss: 0.852, image_val_loss: 0.771, image_train_acc: 0.705, image_val_acc: 0.752, audio_train_loss: 0.428, audio_val_loss: 0.402, audio_train_acc: 0.824, audio_val_acc: 0.849 },
    { epoch: 5, image_train_loss: 0.763, image_val_loss: 0.702, image_train_acc: 0.742, image_val_acc: 0.778, audio_train_loss: 0.381, audio_val_loss: 0.364, audio_train_acc: 0.856, audio_val_acc: 0.871 },
    { epoch: 6, image_train_loss: 0.694, image_val_loss: 0.642, image_train_acc: 0.771, image_val_acc: 0.801, audio_train_loss: 0.345, audio_val_loss: 0.332, audio_train_acc: 0.874, audio_val_acc: 0.888 },
    { epoch: 7, image_train_loss: 0.638, image_val_loss: 0.598, image_train_acc: 0.793, image_val_acc: 0.819, audio_train_loss: 0.315, audio_val_loss: 0.309, audio_train_acc: 0.891, audio_val_acc: 0.902 },
    { epoch: 8, image_train_loss: 0.591, image_val_loss: 0.561, image_train_acc: 0.812, image_val_acc: 0.832, audio_train_loss: 0.292, audio_val_loss: 0.289, audio_train_acc: 0.904, audio_val_acc: 0.914 },
    { epoch: 9, image_train_loss: 0.552, image_val_loss: 0.532, image_train_acc: 0.827, image_val_acc: 0.844, audio_train_loss: 0.271, audio_val_loss: 0.274, audio_train_acc: 0.915, audio_val_acc: 0.921 },
    { epoch: 10, image_train_loss: 0.518, image_val_loss: 0.509, image_train_acc: 0.839, image_val_acc: 0.852, audio_train_loss: 0.254, audio_val_loss: 0.261, audio_train_acc: 0.922, audio_val_acc: 0.928 },
    { epoch: 12, image_train_loss: 0.465, image_val_loss: 0.472, image_train_acc: 0.858, image_val_acc: 0.868, audio_train_loss: 0.228, audio_val_loss: 0.241, audio_train_acc: 0.932, audio_val_acc: 0.935 },
    { epoch: 15, image_train_loss: 0.408, image_val_loss: 0.435, image_train_acc: 0.878, image_val_acc: 0.881, audio_train_loss: 0.198, audio_val_loss: 0.219, audio_train_acc: 0.942, audio_val_acc: 0.942 },
    { epoch: 18, image_train_loss: 0.364, image_val_loss: 0.411, image_train_acc: 0.892, image_val_acc: 0.890, audio_train_loss: 0.174, audio_val_loss: 0.204, audio_train_acc: 0.951, audio_val_acc: 0.948 },
    { epoch: 21, image_train_loss: 0.329, image_val_loss: 0.395, image_train_acc: 0.905, image_val_acc: 0.896, audio_train_loss: 0.155, audio_val_loss: 0.192, audio_train_acc: 0.958, audio_val_acc: 0.952 },
    { epoch: 24, image_train_loss: 0.301, image_val_loss: 0.386, image_train_acc: 0.916, image_val_acc: 0.901, audio_train_loss: 0.139, audio_val_loss: 0.185, audio_train_acc: 0.963, audio_val_acc: 0.954 },
    { epoch: 27, image_train_loss: 0.278, image_val_loss: 0.380, image_train_acc: 0.924, image_val_acc: 0.904, audio_train_loss: 0.126, audio_val_loss: 0.181, audio_train_acc: 0.967, audio_val_acc: 0.957 },
    { epoch: 30, image_train_loss: 0.259, image_val_loss: 0.378, image_train_acc: 0.931, image_val_acc: 0.906, audio_train_loss: 0.116, audio_val_loss: 0.179, audio_train_acc: 0.971, audio_val_acc: 0.958 }
  ],
  splits: {
    test: {
      overall: { accuracy: 0.898, macro_f1: 0.892, weighted_f1: 0.897, loss: 0.384, samples: 1852 },
      per_class: [
        { name: "human", precision: 0.912, recall: 0.884, f1: 0.898, support: 460 },
        { name: "vehicle", precision: 0.886, recall: 0.862, f1: 0.874, support: 382 },
        { name: "animal", precision: 0.924, recall: 0.938, f1: 0.931, support: 570 },
        { name: "empty", precision: 0.865, recall: 0.882, f1: 0.873, support: 440 }
      ],
      confusion_matrix: {
        classes: ["Human", "Vehicle", "Animal", "Empty"],
        matrix: [
          [407, 18, 22, 13],
          [16, 329, 19, 18],
          [12, 14, 535, 9],
          [11, 10, 31, 388]
        ]
      },
      audio_metrics: {
        overall: { accuracy: 0.954, f1: 0.946 },
        per_class: [
          { name: "gunshot", precision: 0.942, recall: 0.931, f1: 0.936, support: 240 },
          { name: "background", precision: 0.961, recall: 0.968, f1: 0.964, support: 485 }
        ],
        confusion_matrix: {
          classes: ["Gunshot", "Background"],
          matrix: [
            [223, 17],
            [15, 470]
          ]
        }
      }
    },
    val: {
      overall: { accuracy: 0.906, macro_f1: 0.901, weighted_f1: 0.905, loss: 0.378, samples: 1850 },
      per_class: [
        { name: "human", precision: 0.920, recall: 0.895, f1: 0.907, support: 458 },
        { name: "vehicle", precision: 0.894, recall: 0.871, f1: 0.882, support: 380 },
        { name: "animal", precision: 0.931, recall: 0.944, f1: 0.937, support: 572 },
        { name: "empty", precision: 0.874, recall: 0.891, f1: 0.882, support: 440 }
      ],
      confusion_matrix: {
        classes: ["Human", "Vehicle", "Animal", "Empty"],
        matrix: [
          [410, 15, 20, 13],
          [14, 331, 18, 17],
          [11, 12, 540, 9],
          [10, 12, 26, 392]
        ]
      },
      audio_metrics: {
        overall: { accuracy: 0.958, f1: 0.951 },
        per_class: [
          { name: "gunshot", precision: 0.949, recall: 0.938, f1: 0.943, support: 242 },
          { name: "background", precision: 0.965, recall: 0.971, f1: 0.968, support: 483 }
        ],
        confusion_matrix: {
          classes: ["Gunshot", "Background"],
          matrix: [
            [227, 15],
            [14, 469]
          ]
        }
      }
    },
    train: {
      overall: { accuracy: 0.931, macro_f1: 0.928, weighted_f1: 0.930, loss: 0.259, samples: 8640 },
      per_class: [
        { name: "human", precision: 0.939, recall: 0.925, f1: 0.932, support: 2140 },
        { name: "vehicle", precision: 0.921, recall: 0.908, f1: 0.914, support: 1780 },
        { name: "animal", precision: 0.952, recall: 0.961, f1: 0.956, support: 2670 },
        { name: "empty", precision: 0.904, recall: 0.915, f1: 0.909, support: 2050 }
      ],
      confusion_matrix: {
        classes: ["Human", "Vehicle", "Animal", "Empty"],
        matrix: [
          [1980, 52, 61, 47],
          [46, 1616, 58, 60],
          [38, 41, 2566, 25],
          [44, 45, 85, 1876]
        ]
      },
      audio_metrics: {
        overall: { accuracy: 0.971, f1: 0.968 },
        per_class: [
          { name: "gunshot", precision: 0.968, recall: 0.955, f1: 0.961, support: 1120 },
          { name: "background", precision: 0.976, recall: 0.982, f1: 0.979, support: 2260 }
        ],
        confusion_matrix: {
          classes: ["Gunshot", "Background"],
          matrix: [
            [1070, 50],
            [40, 2220]
          ]
        }
      }
    }
  },
  dataset_composition: [
    { name: "Caltech Camera Traps (CCT-20)", slug: "cct", samples: 5420, modality: "Image", sensor_type: "Ground Camera Trap (Day/Night)", color: "#5B8266" },
    { name: "BIRDSAI Aerial Thermal", slug: "birdsai", samples: 3840, modality: "Image", sensor_type: "UAV Thermal Infrared (8-14μm)", color: "#D96B43" },
    { name: "Belize Acoustic Array", slug: "belize", samples: 1950, modality: "Audio", sensor_type: "Subtropical Forest Acoustic Array", color: "#F59E0B" },
    { name: "Vietnam Chu Mom Ray", slug: "vietnam", samples: 1680, modality: "Audio", sensor_type: "Dense Rainforest Acoustic Array", color: "#76A284" }
  ],
  calibration: {
    image_temperature: 1.28,
    audio_temperature: 1.15,
    method: "Validation NLL Temperature Scaling (T > 0)",
    ece_before: "8.4%",
    ece_after: "2.1%"
  }
};
