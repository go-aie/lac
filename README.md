# lac

[![Go Reference](https://pkg.go.dev/badge/go-aie/lac/vulndb.svg)][2]

Go Inference API for [LAC][1].


## Installation

1. Install [Paddle Inference Go API][3]
2. Generate [the inference model](cli/README.md#save-inference-model)

    ```bash
    $ python3 cli/cli.py download
    ```
   
3. Install `lac`

    ```bash
    $ go get -u github.com/go-aie/lac
    ```


## Documentation

Check out the [documentation][2].


## Testing

Run tests:

```bash
$ go test -v -race | grep -E 'go|Test'
=== RUN   TestCustomization_Parse
--- PASS: TestCustomization_Parse (0.00s)
=== RUN   TestTriedTree_Search
--- PASS: TestTriedTree_Search (0.00s)
=== RUN   TestLAC_Seg
--- PASS: TestLAC_Seg (0.15s)
ok  	github.com/go-aie/lac	1.188s
```


## License

[MIT](LICENSE)


[1]: https://github.com/baidu/lac
[2]: https://pkg.go.dev/github.com/go-aie/lac
[3]: https://www.paddlepaddle.org.cn/inference/master/guides/install/go_install.html
