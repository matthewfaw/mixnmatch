import kfp
from kfp import dsl
from kfp import gcp


def gcs_download_op(url):
    return dsl.ContainerOp(
        name='GCS - Download',
        image='google/cloud-sdk:216.0.0',
        command=['sh', '-c'],
        arguments=['gsutil cat $0 | tee $1', url, '/tmp/results.txt'],
        file_outputs={
            'data': '/tmp/results.txt',
        }
    )


def echo_op(text):
    return dsl.ContainerOp(
        name='echo',
        image='library/bash:4.4.23',
        command=['sh', '-c'],
        arguments=['echo "$0"', text]
    )

@dsl.pipeline(
    name='Experiment running',
    description='A pipeline to run experiments.'
)
def experiment_running(dataset_bucket,
                       dataset_filename,
                       output_dir,
                       outut_filename,
                       dataset_id="allstate",
                       experiment_type="tree",
                       optimization_budget="constant",
                       otimization_budget_multiplier=100,
                       optimization_budget_height_cap="inf",
                       budget_min=1000,
                       budget_max=3001,
                       budget_step=1000,
                       num_repeats=2,
                       batch_size=50,
                       nu=80,
                       rho=0.9,
                       eta=0.0001,
                       return_best_deepest_node=True,
                       sample_with_replacement=True,
                       mixture_selection_strategy="delaunay-partitioning",
                       columns_to_censor=None):
    v_op = dsl.VolumeOp(
        name="experiment-pvc",
        resource_name="experiment-pvc",
        size="1Gi",
        modes=dsl.VOLUME_MODE_RWM
    )

    download_src = "{}/{}/{}".format(dataset_bucket, dataset_id, dataset_filename)
    download_op = dsl.ContainerOp(
        name="download",
        image="google/cloud-sdk/latest",
        pvolumes={"/transfer": v_op.volume},
        command=["sh", "-c"],
        arguments=["gsutil cp $0 $1", download_src, "/transfer/dataset.p"]
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))

    list_op = dsl.ContainerOp(
        name="list",
        image="alpine:latest",
        pvolumes={"/trans": download_op.pvolume},
        command=["sh", "-c"],
        arguments=["ls /trans"]
    )

    list_op.after(download_op)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(experiment_running, __file__ + '.zip')

