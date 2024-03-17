export default {
  body: `
            <q-tr :props="props">
                <q-td key="id">
                    {{ props.row.id }}
                </q-td>
                <q-td key='set'>
                    <q-btn-group outline>
                        <q-btn outline size="sm" color="blue" round dense
                            @click="props.expand = !props.expand"
                            :icon="props.expand ? 'remove' : 'add'">
                            <q-tooltip>信息</q-tooltip>
                        </q-btn>
                        <q-btn outline size="sm" color="red" round dense icon="delete"
                            @click="() => $parent.$emit('delete', props.row)">
                            <q-tooltip>删除</q-tooltip>
                        </q-btn>
                        <q-btn outline size="sm" color="green" round dense icon="settings"
                            @click="() => $parent.$emit('set', props.row)">
                            <q-tooltip>配置</q-tooltip>
                        </q-btn>
                    </q-btn-group>
                </q-td>
                <q-td key="type">
                    <q-select
                        v-model="props.row.type"
                        :options="''' + str(algo_type_options) + r'''"
                        @update:model-value="() => $parent.$emit('select_type', props.row)"
                        emit-value
                        map-options
                    />
                </q-td>
                <q-td key="spot">
                    <q-select
                        v-model="props.row.spot"
                        :options="''' + str(algo_spot_options) + " [props.row.type]" + r'''"
                        @update:model-value="() => $parent.$emit('select_spot', props.row)"
                        emit-value
                        map-options
                    />
                </q-td>
                <q-td key="algo">
                    <q-select
                        v-model="props.row.algo"
                        :options="''' + str(algo_name_options) + " [props.row.spot]" + r'''"
                        @update:model-value="() => $parent.$emit('select_algo', props.row)"
                        emit-value
                        map-options
                    />
                </q-td>
            </q-tr>
            <q-tr v-show="props.expand" :props="props">
                <q-td colspan="100%">
                    <div class="text-left">This is {{ props.row.algo }} {{ props.row.type }}.</div>
                </q-td>
            </q-tr>
  `,
  data() {
    return {
      value: 0,
    };
  },
  methods: {
    handle_click() {
      this.value += 1;
      this.$emit("change", this.value);
    },
    select_type(){

    },
    select_spot(){

    },
    select_algo(){

    },
    delete(){

    },
    reset() {
      this.value = 0;
    },
  },
  props: {
      algo_type_options: Array,
      algo_spot_options: Array,
      algo_name_options: Array,
      selected_result: Object
  },
};