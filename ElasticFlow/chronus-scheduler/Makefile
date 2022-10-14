rpc:
	python -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/common.proto
	python -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/worker_to_master.proto
	python -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/master_to_worker.proto
	python -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/worker_to_trainer.proto
	python -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/scheduler_to_master.proto
	python -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/trainer_to_scheduler.proto

clean:
	rm -rf runtime/rpc_stubs/*_pb2.py runtime/rpc_stubs/*_pb2_grpc.py
