import ast

class codeTransformer(ast.NodeTransformer):
    def __init__(self):
        self.type_keywords = {
                "DATASET": ["ss", "ts", "ds", "train", "test_data", "dataset", "data"],
                "MODEL_PER": ["accuracy", "loss", "precision", "recall", "f1", "auc", "correct", "incorrect", "results", "error", "y_pred"],
                "MODEL_ARCH": ["model"]
            }
        self.known_apis = ["score"]
    # def visit_Assign(self, node):
    #     targets = node.targets
    #     if len(node.targets) == 1 and isinstance(node.targets[0],ast.Name):
    #         lvalue = node.targets[0].id
    #         print(lvalue)

    def get_base_id(self,node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            return self.get_base_id(node.func)
        elif isinstance(node, ast.Attribute):
            return self.get_base_id(node.value)
        elif isinstance(node, ast.Constant):
            return node.value
        else:
            return None
        
    def determine_type(self, value_name):
        lowered = value_name.lower()
        for type_name, keywords in self.type_keywords.items():
            if any(keyword in lowered for keyword in keywords):
                return type_name
        return "Unknown"
    
    def visit_Assign(self, node):
        arg = node.value
        report_call = ast.Call(
                        func=ast.Name(id='classification_report', ctx = ast.Load()),
                        args = [ast.copy_location(a, arg) for a in arg.args],
                        keywords=[ast.copy_location(ast.keyword(
                                            arg=kw.arg, 
                                            value = ast.copy_location(kw.value, kw.value)
                                            ), kw) for kw in arg.keywords]
                    )
        print(ast.unparse(report_call))
        
    def visit_Expr(self, node):
        
        if isinstance(node.value, ast.Call):
            if node.value.func.attr in self.known_apis:
                print("model_perf")
            else:
                print(ast.dump(node.value))

        # assert_type = None
        # for arg in node.value.args:
        #     if isinstance(arg, ast.Call):
        #         if arg.func.attr == "format":
        #             print(ast.dump(arg.func))
        #             # Check for variable name
        #             for format_arg in arg.args:
                        
        #                 stored_value = self.get_base_id(format_arg).lower()
        #                 print(stored_value)
        #                 assert_type = self.determine_type(stored_value)
                   
        

        # print(stored_value)
    

            # if any(metric.lower() in stored_value for metric in ds_str):
            #     assert_type = "DATASET"
           
    
        # if isinstance(node.value.value, ast.Call):
        #         stored_value = node.value.value.func.value.id.lower()
        # elif isinstance(node.value, ast.Attribute):
        #     stored_value = node.value.value.id
    
        
       
        # var_name = node.value.value.id.lower()
        # if any(ds in var_name for ds in ds_str):
        #     assert_type = "DATASET"
      
       
       

   
        # print(assert_type)
   


# code="""
# print('Found %d incorrect labels' % len(incorrect))
# """
# code ="""
# print('Kfold on RandomForestClassifier: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std()))
# """
code = """
ClassificationReport = classification_report(y_test, y_pred1)
"""
# code = """
# print("Cross validation results:", scores)
# """
tree = ast.parse(code)
transformer = codeTransformer()
tranformed_tree = transformer.visit(tree)
code = ast.unparse(tranformed_tree)
print(code)


