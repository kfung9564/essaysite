from django.db import models
from django.forms import ModelForm


# Create your models here.
class GradeEntry(models.Model):
    owner = models.CharField(max_length=150)
    student_name = models.CharField(max_length=128)
    class_name = models.CharField(max_length=6)
    num_grade = models.DecimalField(decimal_places=1, max_digits=4)
    letter_grade = models.CharField(max_length=1)
    essay = models.TextField()
