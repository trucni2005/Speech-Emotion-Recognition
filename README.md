<h1>Vietnamese Speech Emotion Recognition Model</h1>
<h2>Table of Contents</h2>
<ul>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#model-architecture">Model Architecture</a></li>
  <li><a href="#pipeline">Training and Testing Pipeline</a></li>
  <li><a href="#eng_evaluation_results">English Model Evaluation</a></li>
  <li><a href="#eng_test_results">English Model Test</a></li>
  <li><a href="#vi_evaluation_results">Vietnamese Model Evaluation</a></li>
  <li><a href="#vi_test_results">Vietnamese Model Test</a></li>
</ul>

<h2 id="introduction">Introduction</h2>
<p>This project involves a Voice Emotion Recognition model that detects human emotions from speech signals. The model can identify various emotions such as happiness, sadness, anger, surprised, disgusted, fearful, and neutrality, helping in applications such as customer service, mental health monitoring, and human-computer interaction.</p>

<h2 id="installation">Installation</h2>
<ol>
  <li>Clone the repository:
    <pre><code>git clone https://github.com/trucni2005/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition</code></pre>
  </li>
  <li>Create and activate a virtual environment:
    <pre><code>python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`</code></pre>
  </li>
  <li>Install the required packages:
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
</ol>

<h2 id="usage">Usage</h2>
<ol>
  <li>Download and extract the dataset and models:
    <ol>
      <li>Download the dataset from the following link:
        <p><a href="https://drive.google.com/file/d/1mIqS7K965akbdOnvw5rLtHFZPIeRW3-u/view?usp=drive_link" target="_blank">Download Dataset</a></p>
      </li>
      <li>Download the models from the following link:
        <p><a href="https://drive.google.com/file/d/17-ovCphgXd7xsACBjYe-4-LVeRtsKc9b/view?usp=drive_link" target="_blank">Download Models</a></p>
      </li>
      <li>Extract the dataset and models, and place the files in the root directory of the project (the main folder where your project files are located).</li>
    </ol>
  </li>
  <li>Prepare data - Preprocess data - Extract feature statistics and mel spectrogram images:
    <pre><code>jupyter-notebook audio_feature_pipeline.ipynb</code></pre>
  </li>
  <li>Train CNN model Conv1D using feature statistics:
    <pre><code>jupyter-notebook train_cnn_model_using_feature_statistics.ipynb</code></pre>
  </li>
  <li>Train CNN model Conv2D using mel spectrogram images:
    <pre><code>jupyter-notebook train_cnn_model_using_mel_spectrogram.ipynb</code></pre>
  </li>
  <li>Fine-tune CNN model using feature statistics:
    <pre><code>jupyter-notebook fine_tuning_cnn_model_using_feature_statistics.ipynb</code></pre>
  </li>
  <li>Test the fine-tuned CNN model using feature statistics:
    <pre><code>jupyter-notebook test.ipynb</code></pre>
  </li>
</ol>

<h2 id="model-architecture">Model Architecture</h2>
<img src="/images/model_architecture.png" alt="Model Architecture" width="600">

<h2 id="pipeline">Training and Testing Pipeline</h2>
<img src="/images/pineline.png" alt="Training and Testing Pipeline" width="600">

<h2 id="eng_evaluation_results">English Model Evaluation</h2>
<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Angry</th>
      <th>Disgusted</th>
      <th>Fearful</th>
      <th>Happy</th>
      <th>Neutral</th>
      <th>Sad</th>
      <th>Surprised</th>
      <th>Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MFCC 13</td>
      <td>84.3</td>
      <td>54.1</td>
      <td>47.3</td>
      <td>62.2</td>
      <td>70.1</td>
      <td>70.9</td>
      <td>81.7</td>
      <td>65.6</td>
    </tr>
    <tr>
      <td>MFCC 20</td>
      <td>82.7</td>
      <td>57.0</td>
      <td>52.2</td>
      <td>60.0</td>
      <td>72.6</td>
      <td>72.1</td>
      <td>90.0</td>
      <td>67.3</td>
    </tr>
    <tr>
      <td>MFCC 26</td>
      <td>83.8</td>
      <td>63.4</td>
      <td>44.6</td>
      <td>60.6</td>
      <td>73.9</td>
      <td>74.4</td>
      <td>88.3</td>
      <td>67.7</td>
    </tr>
    <tr>
      <td>MFCC 40</td>
      <td>84.9</td>
      <td>62.2</td>
      <td>51.6</td>
      <td>58.9</td>
      <td>69.4</td>
      <td>68.0</td>
      <td>88.3</td>
      <td>67.0</td>
    </tr>
    <tr>
      <td>ZCR</td>
      <td>74.6</td>
      <td>20.4</td>
      <td>2.7</td>
      <td>18.9</td>
      <td>34.4</td>
      <td>62.8</td>
      <td>3.33</td>
      <td>33.8</td>
    </tr>
    <tr>
      <td>RMS</td>
      <td>71.4</td>
      <td>32.6</td>
      <td>4.0</td>
      <td>26.7</td>
      <td>67.5</td>
      <td>58.1</td>
      <td>55.0</td>
      <td>43.5</td>
    </tr>
    <tr>
      <td>Pitch</td>
      <td>40.5</td>
      <td>39.5</td>
      <td>41.9</td>
      <td>32.8</td>
      <td>41.4</td>
      <td>55.2</td>
      <td>45.0</td>
      <td>41.9</td>
    </tr>
    <tr>
      <td>MFCC 13 + ZCR + RMS + Pitch</td>
      <td>84.3</td>
      <td>58.7</td>
      <td>51.6</td>
      <td>65.0</td>
      <td>75.2</td>
      <td>71.5</td>
      <td>85.0</td>
      <td>68.5</td>
    </tr>
    <tr>
      <td>MFCC 20 + ZCR + RMS + Pitch</td>
      <td>81.6</td>
      <td>59.3</td>
      <td>51.1</td>
      <td>63.3</td>
      <td>74.5</td>
      <td>71.5</td>
      <td>90.0</td>
      <td>68.0</td>
    </tr>
    <tr>
      <td><strong>MFCC 26 + ZCR + RMS + Pitch</strong></td>
      <td><strong>82.2</strong></td>
      <td><strong>59.3</strong></td>
      <td><strong>53.3</strong></td>
      <td><strong>65.0</strong></td>
      <td><strong>75.8</strong></td>
      <td><strong>74.0</strong></td>
      <td><strong>86.7</strong></td>
      <td><strong>69.1</strong></td>
    </tr>
    <tr>
      <td>MFCC 40 + ZCR + RMS + Pitch</td>
      <td>82.2</td>
      <td>58.1</td>
      <td>53.8</td>
      <td>65.6</td>
      <td>72.6</td>
      <td>70.4</td>
      <td>90.0</td>
      <td>68.3</td>
    </tr>
    <tr>
      <td>MSPECT</td>
      <td>77.7</td>
      <td>53.3</td>
      <td>39.6</td>
      <td>49.4</td>
      <td>65.4</td>
      <td>66.5</td>
      <td>76.1</td>
      <td>59.4</td>
    </tr>
  </tbody>
</table>

<h2 id="eng_test_results">English Model Test</h2>

<p>These results indicate that the model performs well in identifying emotions from Vietnamese speech data, with the normalized confusion matrix highlighting the distribution of correctly and incorrectly classified instances for each class.</p>

<h2 id="vi_evaluation_results">Vietnamese Model Evaluation</h2>

<img src="/images/vi_evaluation_confusion_matrix.png" alt="Confusion Matrix" width="600">
<p><b>Figure 3:</b> Raw Confusion Matrix of the Vietnamese Model Evaluation. This matrix shows the number of true positives, false positives, true negatives, and false negatives for each class.</p>

<img src="/images/vi_evaluation_cf_normalized.png" alt="Normalized Confusion Matrix" width="600">
<p><b>Figure 4:</b> Normalized Confusion Matrix of the Vietnamese Model Evaluation. This matrix shows the proportion of true positives, false positives, true negatives, and false negatives for each class.</p>

<p>Overall, the Vietnamese model achieved the following metric on the test set:</p>
<ul>
  <li><b>Accuracy:</b> 81.12%</li>
</ul>

<h2 id="vi_test_results">Vietnamese Model Test</h2>
<img src="/images/vi_test_confusion_matrix.png" alt="Confusion Matrix" width="600">
<p><b>Figure 3:</b> Raw Confusion Matrix of the Vietnamese Model Test. This matrix shows the number of true positives, false positives, true negatives, and false negatives for each class.</p>

<img src="/images/vi_test_cf_normalized.png" alt="Normalized Confusion Matrix" width="600">
<p><b>Figure 4:</b> Normalized Confusion Matrix of the Vietnamese Model Test. This matrix shows the proportion of true positives, false positives, true negatives, and false negatives for each class.</p>

<p>Overall, the Vietnamese model achieved the following metric on the test set:</p>
<ul>
  <li><b>Accuracy:</b> 86.12%</li>
</ul>
