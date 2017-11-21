from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework import viewsets, status
from rest_framework.response import Response
import isFake.services as services


count_vect, tfidf_transformer, clf, articles = services.create_trained_algorithm()


class IsFake(APIView):
    """
    The api endpoint of the characteristics.
    """

    def get(self, request):

        # Prepare the graph
        params = request.query_params
        if len(params) != 1:
            return Response("Please enter a url to prase.", status=status.HTTP_204_NO_CONTENT)
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
