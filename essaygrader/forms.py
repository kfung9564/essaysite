from django import forms


class FileFieldForm(forms.Form):
    file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))


class SubmitEssayForm(forms.Form):
    student_name = forms.CharField(required=True)
    class_name = forms.CharField(required=True)
    essay_text = forms.CharField(required=True, widget=forms.Textarea)
