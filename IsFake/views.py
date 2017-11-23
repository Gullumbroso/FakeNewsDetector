from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework import viewsets, status
from rest_framework.response import Response
import FakeNewsServer.IsFake.services as services


# Train the model first thing when the server starts running
count_vect, tfidf_transformer, clf, articles = services.get_trained_machine()


class IsFake(APIView):
    """
    The api endpoint for validating the received article.
    """

    def get(self, request):

        params = request.query_params
        if len(params) != 1:
            return Response("Please enter a url to parse.", status=status.HTTP_204_NO_CONTENT)
        else:
            url = params['article_url']
            article_title, article_content = services.parse_article(url)
            whole_content = article_title + '\n' + article_content
            prediction, score = services.predict(count_vect, tfidf_transformer, clf, articles, whole_content)
            response = {
                'article_title': article_title,
                'article_content': article_content,
                'prediction': prediction,
                'confidence': score
            }

            return Response(response, status=status.HTTP_200_OK)
