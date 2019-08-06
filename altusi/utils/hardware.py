from mxnet import context, gluon


def get_gpu(i=0):
    return context.gpu(i) if context.num_gpus() > i else context.cpu()

def get_all_gpus():
    ctxes = [context.gpu(i) for i in range(context.num_gpus())]
    return ctxes if ctxes else [context.cpu()]

def split_batch(X, y, ctx_list):
    assert X.shape[0] == y.shape[0]

    return (gluon.utils.split_and_load(X, ctx_list),
            gluon.utils.split_and_load(y, ctx_list))
