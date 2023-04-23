#!/bin/bash
usage() {
     echo "Usage: ${0} [-t|--type] [-img1] [-img2]]" 1>&2
     exit 1
}

ROT=0
while [[ $# -gt 0 ]];do
     key=${1}
     case ${key} in
          -t|--type)
          TYPE=${2}
          echo "TYPE : $TYPE"
          shift 2
          ;;
          -img1)
          IMG1=${2}
          echo "IMAGE 1 : $IMG1"
          shift 2
          ;;
          -img2)
          IMG2=${2}
          echo "IMAGE 2 : $IMG2"
          shift 2
          ;;
          *)
          usage
          shift
          ;;
     esac
done

if [[ $TYPE == "feature_extraction" ]];
then
     python3 src/test_feature_extraction.py --img $IMG1 #--rotate
elif [[ $TYPE == "feature_matching" ]];
then
     python3 src/test_feature_matching.py --img1 $IMG1 --img2 $IMG2 #--rotate
elif [[ $TYPE == "roi_matching" ]];
then
     python3 src/test_ROIs_matching.py --img1 $IMG1 --img2 $IMG2 #--rotate
elif [[ $TYPE == "stitch" ]];
then
     python3 src/test_stitch.py --img1 $IMG1 --img2 $IMG2 #--rotate
fi