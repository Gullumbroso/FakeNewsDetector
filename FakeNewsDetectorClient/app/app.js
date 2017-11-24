/**
 * Created by gullumbroso on 07/12/2016.
 */

angular.module('FakeNews', ['ngMaterial'])

    .config(function ($mdThemingProvider) {
        $mdThemingProvider.theme('default')
            .primaryPalette('blue-grey', {
                'default': '900'
            })
            .accentPalette('pink', {
                'default': '500'
            })
    })

    .run([
        function () {

            // global constants

        }])

    .controller('FakeNewsController', function ($scope, $http, $timeout, $mdDialog) {

        var DEFAULT_BUTTON_TITLE = "Start";
        var AGAIN_BUTTON_TITLE = "New Article";
        var AWAITING_INPUT_STATUS = "awaiting";
        var LOADING_STATUS = "loading";
        var ANSWERED_STATUS = "answered";
        var BASE_URL = "http://gullumbroso.pythonanywhere.com";
        var APP_PATH = "/is_fake/";

        $scope.query = {};
        $scope.noCache = true;

        $scope.status = AWAITING_INPUT_STATUS;
        $scope.goTitle = DEFAULT_BUTTON_TITLE;
        $scope.tryAgainTitle = AGAIN_BUTTON_TITLE;
        $scope.presentResults = false; // For animation;
        $scope.content = "";
        $scope.answer = "";
        $scope.article = {
            title: "",
            content: ""
        };
        $scope.selectedItem = null;
        $scope.sourceText = null;
        $scope.destinationText = null;
        $scope.query.url = "";

        // $scope.querySearch = function (text) {
        //     if ($scope.nodes.length == 0) {
        //         $http.get(base_url + '/characteristics/')
        //             .then(function (response) {
        //                 $scope.nodes = response.data.results;
        //                 return text ? $scope.nodes.filter(createFilterFor(text)) : $scope.nodes;
        //             })
        //     } else {
        //         return text ? $scope.nodes.filter(createFilterFor(text)) : $scope.nodes;
        //     }
        // };
        //
        // function createFilterFor(query) {
        //     var lowercaseQuery = angular.lowercase(query);
        //
        //     return function filterFn(state) {
        //         return (state.value.indexOf(lowercaseQuery) === 0);
        //     };
        //
        // }

        function presentAlertDialog(title, content) {
            $mdDialog.show(
                $mdDialog.alert()
                    .parent(angular.element(document.body))
                    .clickOutsideToClose(true)
                    .title(title)
                    .textContent(content)
                    .ariaLabel('Alert Dialog')
                    .ok("OK")
                    .targetEvent(ev)
            );
        }

        $scope.go = function (event) {

            if ($scope.status == AWAITING_INPUT_STATUS) {

                if ($scope.query.url == "" || $scope.status == LOADING_STATUS) return;
                var url = BASE_URL + APP_PATH;
                var params = {
                    article_url: $scope.query.url
                };
                $scope.status = LOADING_STATUS;
                $http.get(url, {params: params})
                    .then(function (response) {
                            // success
                            $scope.prediction = response.data['prediction'];
                            $scope.article.title = response.data['article_title'];
                            $scope.article.content = response.data['article_content'];
                            $scope.confidence_score = response.data['confidence'].toFixed(4);

                            if (Math.abs($scope.confidence_score) < 0.1) {
                                $scope.confidence = "Not confident."
                            } else if (Math.abs($scope.confidence_score) < 0.2) {
                                $scope.confidence = "Not very confident..."
                            } else if (Math.abs($scope.confidence_score) < 0.45) {
                                $scope.confidence = "Pretty confident..."
                            } else {
                                $scope.confidence = "Very confident."
                            }

                            if ($scope.prediction == 'fake_news') {
                                $scope.resultIcon = "assets/StarTypeLogo_fake_alert.png";
                                $scope.answer = "This is FAKE!";
                            } else if ($scope.prediction == 'real_news') {
                                $scope.resultIcon = "assets/StarTypeLogo_real_alert.png";
                                $scope.answer = "This is real news.";
                            } else {
                                $scope.answer = $scope.prediction;
                            }
                            $scope.status = ANSWERED_STATUS;
                            $scope.goTitle = AGAIN_BUTTON_TITLE;
                            $timeout(function () {
                                $scope.presentResults = true;
                            }, 500);
                        },
                        function (err) {
                            // error
                            $scope.status = ANSWERED_STATUS;
                            $scope.answer = "Couldn't parse the article, please try again.";
                            $scope.resultIcon = "assets/StarTypeLogo_PROBLEM.png";
                            $scope.confidence = "";
                            $scope.confidence_score = "";
                            $scope.url = "";
                            $scope.goTitle = AGAIN_BUTTON_TITLE;
                            $timeout(function () {
                                $scope.presentResults = true;
                            }, 500);
                        });
            }
        };

        $scope.tryAgain = function (event) {
            $scope.status = AWAITING_INPUT_STATUS;
            $scope.goTitle = DEFAULT_BUTTON_TITLE;
            $scope.presentResults = false;
        }
    });
