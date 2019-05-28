#!/bin/bash

set -o nounset
set -o errexit

VERBOSE_MODE=0

function error_handler()
{
  local STATUS=${1:-1}
  [ ${VERBOSE_MODE} == 0 ] && exit ${STATUS}
  echo "Exits abnormally at line "`caller 0`
  exit ${STATUS}
}
trap "error_handler" ERR

PROGNAME=`basename ${BASH_SOURCE}`
DRY_RUN_MODE=0

function print_usage_and_exit()
{
  set +x
  local STATUS=$1
  echo "Usage: ${PROGNAME} [-v] [-v] [--dry-run] [-h] [--help]"
  echo ""
  echo " Options -"
  echo "  -v                 enables verbose mode 1"
  echo "  -v -v              enables verbose mode 2"
  echo "      --dry-run      show what would have been dumped"
  echo "  -h, --help         shows this help message"
  exit ${STATUS:-0}
}

function debug()
{
  if [ "$VERBOSE_MODE" != 0 ]; then
    echo $@
  fi
}

#GETOPT=`getopt -o vh --long dry-run,help -n "${PROGNAME}" -- "$@"`
GETOPT=`getopt vh $*`
if [ $? != 0 ] ; then print_usage_and_exit 1; fi

eval set -- "${GETOPT}"

while true
do case "$1" in
     -v)            let VERBOSE_MODE+=1; shift;;
     --dry-run)     DRY_RUN_MODE=1; shift;;
     -h|--help)     print_usage_and_exit 0;;
     --)            shift; break;;
     *) echo "Internal error!"; exit 1;;
   esac
done

if (( VERBOSE_MODE > 1 )); then
  set -x
fi

if [ ${#} != 0 ]; then print_usage_and_exit 1; fi

set -o errexit
function readlink()
{
    TARGET_FILE=$2
    cd `dirname $TARGET_FILE`
    TARGET_FILE=`basename $TARGET_FILE`

    # Iterate down a (possible) chain of symlinks
    while [ -L "$TARGET_FILE" ]
    do
        TARGET_FILE=`readlink $TARGET_FILE`
        cd `dirname $TARGET_FILE`
        TARGET_FILE=`basename $TARGET_FILE`
    done

    # Compute the canonicalized name by finding the physical path
    # for the directory we're in and appending the target file.
    PHYS_DIR=`pwd -P`
    RESULT=$PHYS_DIR/$TARGET_FILE
    echo $RESULT
}
export -f readlink

# current dir of this script
CDIR=$(readlink -f $(dirname $(readlink -f ${BASH_SOURCE[0]})))
PDIR=$(readlink -f $(dirname $(readlink -f ${BASH_SOURCE[0]}))/..)


rm -rf ./output
mkdir -p output/result_dir
rm -rf data.conf

lowercase='False'
#bert_model_dir=${CDIR}/cased_L-12_H-768_A-12
bert_model_dir=..Models/bert

#lowercase='True'
#bert_model_dir=${CDIR}/uncased_L-12_H-768_A-12
#bert_model_dir=${CDIR}/engwiki.1m-step.uncased_L-12_H-768_A-12

python DeId_bert.py   \
        --task_name="NER"  \
        --do_train=True   \
        --do_predict=True \
        --use_crf=True \
        --data_dir=../Datasets/data  \
        --vocab_file=${bert_model_dir}/vocab.txt  \
        --do_lower_case=${lowercase} \
        --bert_config_file=${bert_model_dir}/bert_config.json \
        --init_checkpoint=${bert_model_dir}/bert_model.ckpt   \
        --max_seq_length=180   \
        --lstm_size=256 \
        --train_batch_size=16   \
        --eval_batch_size=32   \
        --predict_batch_size=32   \
        --bert_dropout_rate=0.2 \
        --bilstm_dropout_rate=0.2 \
        --learning_rate=2e-5   \
        --num_train_epochs=100   \
        --data_config_path=${CDIR}/data.conf \
--output_dir=${CDIR}/output/result_dir/