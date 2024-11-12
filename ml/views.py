# sentiment_app/views.py
from django.shortcuts import render
from. models import predict_sentiment

def sentiment_analysis(request):
    result = None
    if request.method == "POST":
        text = request.POST.get("text")
        result = predict_sentiment(text)
    return render(request, 'sentiment.html', {"result": result})
