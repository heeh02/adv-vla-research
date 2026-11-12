"""Return an independent overlay candidate; never edits the original path."""
import hashlib
ORIGINAL_SHA='fdcaebdf7bb19fbe6401ea6055a465fc6064d124f3821c08d77edacf69cf00f6'
def build(raw):
 if type(raw)is not bytes or hashlib.sha256(raw).hexdigest()!=ORIGINAL_SHA:raise ValueError('PVP_original_Gemma_SHA')
 old=b"SCAN_NATIVE_FIELDS = ('receiver_residual_input', 'hidden_post_block',"
 new=b"SCAN_NATIVE_FIELDS = ('receiver_residual_input', 'hidden_after_attention_residual', 'hidden_post_block',"
 if raw.count(old)!=1:raise ValueError('PVP_one_existing_SCAN_field_tuple')
 out=raw.replace(old,new);compile(out,'<PVP-independent-gemma-overlay>','exec')
 return out
