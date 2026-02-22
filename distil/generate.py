import torch
from transformers import AutoTokenizer

from distil.student import create_student
from distil.teacher import load_teacher
from sft.load_tinyllama import MODEL_DIR

device = "mps"
prompt = "The capital of France is"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "models/tinyllama-1.1b")
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Teacher
teacher = load_teacher()
teacher.to(device)
teacher_out = teacher.generate(input_ids, max_new_tokens=20)
print(f"Teacher: {tokenizer.decode(teacher_out[0])}")

# Student
student = create_student(vocab_size=32000)
student.load_state_dict(torch.load("student.pt", map_location=device))
student.to(device)
student.eval()
student_out = student.generate(input_ids, max_new_tokens=20)
print(f"Student: {tokenizer.decode(student_out[0])}")
