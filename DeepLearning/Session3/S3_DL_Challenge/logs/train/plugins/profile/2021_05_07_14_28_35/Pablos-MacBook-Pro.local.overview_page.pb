?	+???w@+???w@!+???w@	ϋ??
?@ϋ??
?@!ϋ??
?@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:+???w@q=
ףp??Ay?&1,@Y?v??/??rEagerKernelExecute 0*	     @a@2U
Iterator::Model::ParallelMapV2L7?A`???!???7??G@)L7?A`???1???7??G@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9??v????!??\;0?B@)9??v????1??\;0?B@:Preprocessing2F
Iterator::Model??~j?t??!?%~F?K@){?G?z??1?(?3J?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?|?!??(?3J@)y?&1?|?1??(?3J@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!???,d@)?~j?t?h?1???,d@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ϋ??
?@I??k?jX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	q=
ףp??q=
ףp??!q=
ףp??      ??!       "      ??!       *      ??!       2	y?&1,@y?&1,@!y?&1,@:      ??!       B      ??!       J	?v??/???v??/??!?v??/??R      ??!       Z	?v??/???v??/??!?v??/??b      ??!       JCPU_ONLYYϋ??
?@b q??k?jX@Y      Y@q?e?ړ???"?
both?Your program is POTENTIALLY input-bound because 9.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 