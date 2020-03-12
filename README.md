
## How to run

```bash
LOGDIR='<directory to use for storing train logs>'
CONFIG='<path to .yml with config>'

catalyst-dl run --expdir src \
    --config ${CONFIG} \
    --logdir ${LOGDIR} \
    --distributed \
    --no-apex \
    --verbose
```