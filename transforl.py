from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
# import modelscope.models.nlp.csanmt.translation.CsanmtForTranslation 
input_sequence = 'Elon Musk, co-founder and chief executive officer of Tesla Motors.'

pipeline_ins = pipeline(task=Tasks.translation, model="damo/nlp_csanmt_translation_en2zh")
outputs = pipeline_ins(input=input_sequence)
sess = pipeline_ins._session
# print(outputs['translation']) 



