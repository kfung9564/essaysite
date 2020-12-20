from django.shortcuts import render

# Create your views here.
from essaygrader.models import GradeEntry


def view_gradebook(request, *args, **kwargs):
    query = GradeEntry.objects.filter(owner=request.user.username).order_by('class_name')

    return render(request, "gradebook.html", {"entries": query})