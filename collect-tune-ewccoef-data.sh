#!/bin/bash
scp -r -P 23333 $PYUNRAMWK04:/data_shared/Docker/pyun/IncDet3/incdet3/logs/20200907-* logs/20200907-tune-ewccoef-logs/
scp -r -P 23333 $PYUNRAMWK06:/data_shared/Docker/pyun/IncDet3/incdet3/logs/20200907-* logs/20200907-tune-ewccoef-logs/
scp -r -P 23333 $PYUNRAMWK03:/data_shared/Docker/pyun/IncDet3/incdet3/logs/20200907-* logs/20200907-tune-ewccoef-logs/
scp -r -P 23333 $PYUNRAMWK04:/data_shared/Docker/pyun/IncDet3/incdet3/saved_weights/20200907-* saved_weights/20200907-tune-ewccoef-saved_weights/
scp -r -P 23333 $PYUNRAMWK06:/data_shared/Docker/pyun/IncDet3/incdet3/saved_weights/20200907-* saved_weights/20200907-tune-ewccoef-saved_weights/
scp -r -P 23333 $PYUNRAMWK03:/data_shared/Docker/pyun/IncDet3/incdet3/saved_weights/20200907-* saved_weights/20200907-tune-ewccoef-saved_weights/