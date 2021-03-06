import io
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix ,roc_curve, auc, accuracy_score
from itertools import cycle

class MyEstimator():
    def __init__(self):
        pass

    def get_img_from_fig(self, fig, dpi=100):                       # plt 轉為 numpy
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.close('all')
        return img

    def confusion(self, y_true, y_pred, val_keyLabel=None, classes=2, logPath=None, mode = ''):
        Accuracy = accuracy_score(y_true, y_pred)
        Specificity = 0.0
        Sensitivity = 0.0
        plot_img_np = None

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        error_list = {'1_to_0':[], '0_to_1':[]}
        t1p0 = []
        t0p1 = []
        if val_keyLabel:
            for i in range(len(val_keyLabel)):

                if y_true[i] == 1 and y_pred[i] == 0:
                    t1p0.append(val_keyLabel[i])

                elif y_true[i] == 0 and y_pred[i] == 1:
                    t0p1.append(val_keyLabel[i])

        error_list['1_to_0'] = t1p0
        error_list['0_to_1'] = t0p1
        
        if len(classes) == 2:
            cf_matrix = confusion_matrix(y_true, y_pred)
            TP, FN, FP, TN = confusion_matrix(y_true, y_pred).ravel()
            if (TN + FP) != 0:
                Specificity = TN / (TN + FP)
            if (TP + FN) != 0:
                Sensitivity = TP / (TP + FN)
        else:
            cf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        if logPath:
            plt.close("all")
            df_cm = pd.DataFrame(cf_matrix, classes, classes)
            plt.figure(figsize = (9,6))
            sns.set(font_scale=2)
            sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
            plt.xlabel('Predict', fontsize=20)        
            plt.ylabel('True', fontsize=20)
            plt.title(str(mode) + '_Acc : {:.2f} | Specificity : {:.2f} | Sensitivity : {:.2f}'.format(Accuracy, Specificity, Sensitivity), fontsize=20)

            plot_img_np = self.get_img_from_fig(plt)    # plt 轉為 numpy]
            plt.savefig(logPath + "//" + str(mode) + "_confusion.jpg", bbox_inches='tight')
            plt.close('all')


        return Accuracy, Specificity, Sensitivity, error_list, plot_img_np

    def compute_auc(self, y_true, y_score, classes, logPath=None, mode = ''):
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

        fpr, tpr, roc_auc = dict(), dict(), dict()
        y_label = np.eye(len(classes))[y_true]
        y_label = y_label.reshape(len(y_true), len(classes))
        y_score = np.array(y_score)
        plot_img_np = None

        if len(y_label) > 1 :
            for i in range(len(classes)):
                fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i], pos_label=1)
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot all ROC curves
            if logPath:
                plt.close("all")
                sns.set(font_scale=1)
                plt.figure()
                
                for i, color in zip(range(len(classes)), colors):
                    plt.plot(fpr[i], tpr[i], color=color, lw = 2,
                            label='ROC curve of level {0} (area = {1:0.2f})'
                            ''.format(i + 1, roc_auc[i]))
                
                plt.plot([0, 1], [0, 1], 'k--', lw = 2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(str(mode) + '_multi-calss ROC')
                plt.legend(loc="lower right")

                plot_img_np = self.get_img_from_fig(plt)    # plt 轉為 numpy
                plt.savefig(logPath + "//" + str(mode) +"_compute_auc.jpg", bbox_inches='tight')
                plt.close("all")
        else:
            roc_auc = -1.00

        return roc_auc, plot_img_np
