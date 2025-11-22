**TODO**
- capire se voglio tenere @classmethod in pansharpening_metrics/metrics.py/MetricsConfig
- capire se per varie metriche mi servono sempre non-overlapping windows!
- downsampling function da mettere in pansharpening_metrics/downsampling.py, sia per pan che sharp to lr -- **DONE**
- aggiungere ref giusta di downsampling function in metrics.py -- **DONE**

- FINIRE DI RICONTROLLARE MAIN.PY E SETUP.PY, per il resto dovremmo esserci: mancano i test

- test banale fatto e sembra andare, ricontrollare un po' gli shapes e capire se manca qualcosa. 


**NOTES**
- preprocess_for_metrics spostato in pansharpening_metrics/utils.py
- i resize di sharp e pan per calcolare le metriche con lr: dentro alle funzioni delle metriche --- **DONE**
  o li metto prima? Dentro LocalCluster e Client??? <-- **CAPIRE** ---> MESSO FUORI PRIMA DI DASK