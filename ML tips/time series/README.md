## Time Series classification

### Fastai extensions for timeseries:

https://github.com/timeseriesAI/tsai

https://github.com/tcapelle/TimeSeries_fastai 

https://forums.fast.ai/t/time-series-sequential-data-study-group/29686/331

https://github.com/ai-fast-track/timeseries

### Data augmentation for timeseries:
https://medium.com/@keur.plkar/from-sound-to-image-to-building-an-image-classifier-for-sound-with-fast-ai-3294909b3885

https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6

**Note**: may be very slow, for pure timeseires classification better to use this: https://github.com/timeseriesAI/tsai/blob/master/tutorial_nbs/03_Time_Series_Transforms.ipynb

## Time series forecasting
### N-BEATS
article https://towardsdatascience.com/n-beats-beating-statistical-models-with-neural-nets-28a4ba4a4de8
paper https://arxiv.org/abs/1905.10437

### ETSformer: Exponential Smoothing Transformers for Time-series Forecasting
https://arxiv.org/abs/2202.01381

pytorch implementation: https://github.com/lucidrains/ETSformer-pytorch

official github: https://github.com/salesforce/ETSformer

### Do We Really Need Deep Learning Models for Time Series Forecasting?

https://arxiv.org/abs/2101.02118

Followup discussion: https://www.reddit.com/r/MachineLearning/comments/t9ou4z/d_do_we_really_need_deep_learning_models_for_time/
- more modern dl forecasting methods eg temporal fusion transformer (TFT) may still be better
- GBDT by default cannot extrapolate so may also fail in longer term forecasting with trend (eg weeks months years)
  - will need to de-trend to make timeseries stationary (may not always be possible?)
  
 
