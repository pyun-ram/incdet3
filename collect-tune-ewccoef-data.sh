#!/bin/bash
scp -r -P 23333 $PYUNRAMWK04:/data_shared/Docker/pyun/IncDet3/incdet3/logs/*kitti4+1* logs/20200907-expkitti4+1-ewc-logs/
scp -r -P 23333 $PYUNRAMWK06:/data_shared/Docker/pyun/IncDet3/incdet3/logs/*kitti4+1* logs/20200907-expkitti4+1-ewc-logs/
scp -r -P 23333 $PYUNRAMWK03:/data_shared/Docker/pyun/IncDet3/incdet3/logs/*kitti4+1* logs/20200907-expkitti4+1-ewc-logs/
scp -r -P 23333 $PYUNRAMWK04:/data_shared/Docker/pyun/IncDet3/incdet3/saved_weights/*kitti4+1* saved_weights/20200907-expkitti4+1-ewc-saved_weights/
scp -r -P 23333 $PYUNRAMWK06:/data_shared/Docker/pyun/IncDet3/incdet3/saved_weights/*kitti4+1* saved_weights/20200907-expkitti4+1-ewc-saved_weights/
scp -r -P 23333 $PYUNRAMWK03:/data_shared/Docker/pyun/IncDet3/incdet3/saved_weights/*kitti4+1* saved_weights/20200907-expkitti4+1-ewc-saved_weights/