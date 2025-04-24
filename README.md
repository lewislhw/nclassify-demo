

# TF-IDF + Linear SVC Demo
快速体验“通用文本 N 分类”

---

## 1. 功能简介
- **输入**：任意 CSV 文件  
  - 指定哪一列是 **文本**（对话、评论等）  
  - 指定哪一列是 **标签**（类别名，枚举类）  
- **流程**  
  1. 结巴分词 → TF-IDF 向量  
  2. 线性 SVM (`LinearSVC`) 训练  
  3. 80/20 （可调）切分并输出分类报告  
  4. 保存词表 / IDF / 训练好的模型  
- **输出**：  
  ```
  demo_model/
      ├── tfidf_vocab.json   # 词 -> 列索引
      ├── tfidf_idf.npy      # IDF 数值
      └── linear_svc.joblib  # 训练好的 LinearSVC
  ```

---

## 2. 依赖安装
```bash
conda create -n textclf python=3.9 -y
conda activate textclf
pip install -r requirements.txt
```

`requirements.txt` 
```
pandas
scikit-learn
jieba
joblib
numpy
```

---

## 3. 快速使用

### 3.1 有表头 CSV
```bash
python demo_tfidf_svc.py \
  --csv df_file.csv \
  --text-col 0 \
  --label-col 1 \
  --test-size 0.25
```
数据集示例 df_file.csv 有表头，第一列为文本，第二列为标签，测试数据集比例为 0.25

### 3.2 无表头 CSV
```bash
python demo_tfidf_svc.py \
  --csv df_file.csv \
  --text-col 0 \
  --label-col 1 \
  --no-header
```

参数说明：

| 参数              | 说明             | 默认     |
|-----------------|----------------|--------|
| `--csv`         | 输入 CSV 路径      | **必填** |
| `--text-col`    | 文本列索引（0 起）     | `0`    |
| `--label-col`   | 标签列索引（0 起）     | `1`    |
| `--no-header`   | 无表头（CSV 首行是数据） | 否      |
| `--test-size`   | 测试集占比          | `0.2`  |
| `--random-seed` | 随机种子           | `42`   |

---

## 4. 典型输出

```text
>>> 样例 text / label
[621] Newcastle to join Morientes race
 
 Newcastle have joined th ...  ==>  1
[1234] Savvy searchers fail to spot ads
 
 Internet search engine u ...  ==>  2
[571] Wenger signs new deal
 
 Arsenal manager Arsene Wenger has s ...  ==>  1
------------------------------------------------------------
Building prefix dict from the default dictionary ...
Dumping model to file cache /tmp/jieba.cache
Loading model cost 0.462 seconds.
Prefix dict has been built successfully.

=== 分类报告 ===
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       104
           1       0.98      1.00      0.99       128
           2       0.96      0.98      0.97       100
           3       0.97      0.99      0.98        97
           4       1.00      0.95      0.98       128

    accuracy                           0.98       557
   macro avg       0.98      0.98      0.98       557
weighted avg       0.98      0.98      0.98       557

模型文件已保存到 demo_model/

```

