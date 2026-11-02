const RESULTS = {
  "test": [
    {
      "Model": "QDA",
      "Accuracy": 0.9426,
      "Precision": 0.9617,
      "Recall": 0.9454,
      "F1": 0.9535,
      "F1_Rejected": 0.9252,
      "F1_Macro": 0.9393,
      "F1_Weighted": 0.9428,
      "ROC_AUC": 0.9806
    },
    {
      "Model": "LDA",
      "Accuracy": 0.9239,
      "Precision": 0.9533,
      "Recall": 0.9228,
      "F1": 0.9378,
      "F1_Rejected": 0.902,
      "F1_Macro": 0.9199,
      "F1_Weighted": 0.9242,
      "ROC_AUC": 0.9737
    },
    {
      "Model": "Logistic Regression",
      "Accuracy": 0.9145,
      "Precision": 0.921,
      "Recall": 0.9435,
      "F1": 0.9321,
      "F1_Rejected": 0.8847,
      "F1_Macro": 0.9084,
      "F1_Weighted": 0.9142,
      "ROC_AUC": 0.9726
    }
  ],
  "cv": [
    {
      "Model": "QDA",
      "CV_accuracy_mean": 0.9329,
      "CV_accuracy_std": 0.0112,
      "CV_precision_mean": 0.9559,
      "CV_precision_std": 0.0111,
      "CV_recall_mean": 0.9355,
      "CV_recall_std": 0.013,
      "CV_f1_mean": 0.9455,
      "CV_f1_std": 0.0092,
      "CV_f1_macro_mean": 0.9291,
      "CV_f1_macro_std": 0.0117,
      "CV_roc_auc_mean": 0.9732,
      "CV_roc_auc_std": 0.0071
    },
    {
      "Model": "LDA",
      "CV_accuracy_mean": 0.9212,
      "CV_accuracy_std": 0.0142,
      "CV_precision_mean": 0.959,
      "CV_precision_std": 0.0105,
      "CV_recall_mean": 0.9125,
      "CV_recall_std": 0.0156,
      "CV_f1_mean": 0.9351,
      "CV_f1_std": 0.0119,
      "CV_f1_macro_mean": 0.9175,
      "CV_f1_macro_std": 0.0147,
      "CV_roc_auc_mean": 0.966,
      "CV_roc_auc_std": 0.0078
    },
    {
      "Model": "Logistic Regression",
      "CV_accuracy_mean": 0.9142,
      "CV_accuracy_std": 0.0163,
      "CV_precision_mean": 0.9291,
      "CV_precision_std": 0.0179,
      "CV_recall_mean": 0.9336,
      "CV_recall_std": 0.0137,
      "CV_f1_mean": 0.9313,
      "CV_f1_std": 0.0129,
      "CV_f1_macro_mean": 0.9085,
      "CV_f1_macro_std": 0.0176,
      "CV_roc_auc_mean": 0.9663,
      "CV_roc_auc_std": 0.0082
    }
  ]
};
